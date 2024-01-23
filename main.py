import os
import torch
import json

from models import getModel
from optimizer import getOptimizer
from utility.loss import smooth_crossentropy
from utility.data import DataLoader
from utility.dataLogger import DataLogger
from utility.utils import initialize
from utility.LRScheduler import getLRScheduler, _LRScheduler
from utility.rhoScheduler import getRhoScheduler, _rhoScheduler
from utility.modelSaver import ModelSaver
from utility.args import Args


"""
run:
    python -m torch.distributed.run main.py
"""

Args.add_argument("--logDir", type=str, help="main directory to store logs")
Args.add_argument("--logSubDir", type=str, help="subdir in logDir to store logs for this run")
Args.add_argument("--epochs", type=int, help="Total number of epochs")
Args.add_argument("--contin", type=bool, help="Whether to continue from checkpoint. In continue mode parameters are read from params.json file, input file is ignored.")

if __name__ == "__main__":
    Args.parse_args()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    logDir = os.path.join(Args.logDir, Args.logSubDir)

    if Args.contin:
        with open(os.path.join(logDir, "params.json"), "r") as file:
            parameters = json.load(file)
        Args.parse_args_contin(parameters)


    torch.distributed.init_process_group(backend="nccl", init_method="env://", rank = int(os.getenv("SLURM_PROCID", -1))) #set rank to 'SLURM_PROCID' if started with slurm, else to -1
    local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    initialize() # set up seed and cudnn

    if torch.distributed.get_rank() == 0:
        os.makedirs(logDir, exist_ok=True)
        with open(os.path.join(logDir, "params.json"), "w") as file:
            json.dump(vars(Args.data), file, indent = 4)

    dataLogger = DataLogger()
    dataset = DataLoader()

    model = getModel()(num_classes=dataset.numClasses)
    model = model.cuda(local_rank)
    if hasattr(torch, "compile"):
         model = torch.compile(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model)


    optimizer = getOptimizer(model.parameters())
    lrScheduler: _LRScheduler = getLRScheduler(optimizer)
    rhoScheduler: _rhoScheduler = getRhoScheduler(optimizer)
    modelSaver = ModelSaver(model = model, optimizer = optimizer)
    
    startEpoch = 1
    if Args.contin:
        startEpoch = modelSaver.loadModel("checkpoint.model")
        startEpoch += 1
        model = model.cuda(local_rank)
        if startEpoch >= Args.epochs:
            raise RuntimeError(f"Can't continue model from epoch {startEpoch} to max epoch {Args.epochs}.")
    else:
        modelSaver(0)

    torch.distributed.barrier() #wait until all workers are done with initialization
    dataLogger.printHeader()
    state = {
        "model": model,
        "lrScheduler": lrScheduler,
        "optimizer": optimizer,
    }
    for epoch in range(startEpoch, Args.epochs+1):
        dataset.train.sampler.set_epoch(epoch)

        model.train()
        numBatches = len(dataset.train)
        dataLogger.startTrain(trainDataLen = numBatches)

        if Args.optimizer in ["MSAM","AdamW_MSAM"]:
            optimizer.move_up_to_momentumAscent()
        for i, batch in enumerate(dataset.train):
            lrScheduler.step(epoch-1, (i+1)/numBatches)
            rhoScheduler.step(epoch-1, (i+1)/numBatches)

            inputs, targets = (b.cuda(local_rank) for b in batch)

            # SAM needs to calculate second forward/backward paths, so target, inputs, and model have to be passed to optimizer. 
            # we are not doing this for MSAM so that MSAM can be used as a drop in replacement for SGD/AdamW.
            if Args.optimizer in ["SAM", "AdamW_SAM", "ESAM","lookSAM"]:
                loss, predictions = optimizer.step(model, inputs, targets)
            else:
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                loss.mean().backward()
                if Args.grad_clip_norm != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)
                optimizer.step()

            optimizer.zero_grad()

            with torch.no_grad():
                state["loss"] = loss
                state["predictions"] = predictions
                state["targets"] = targets
                dataLogger(state)

        dataLogger.flush()
        

        if Args.optimizer in ["MSAM","AdamW_MSAM"]:
            optimizer.move_back_from_momentumAscent()
        dataLogger.startTest()
        model.eval()
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.cuda(local_rank) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                state["loss"] = loss
                state["predictions"] = predictions
                state["targets"] = targets
                dataLogger(state)

            dataLogger.flush()
            modelSaver(epoch)

    # run extra epoch (forward path only) to calculate batch norm statistics (not needed for ViT since no BN-layers are used)
    # probably a few iterations would be sufficient until BN-statistics converge, but we didnt test this
    if Args.optimizer in ["MSAM","AdamW_MSAM"] and Args.model != "ViT":
        with torch.no_grad():
            model.train()
            numBatches = len(dataset.train)
            dataLogger.startTrain(trainDataLen = numBatches)

            for i, batch in enumerate(dataset.train):
                inputs, targets = (b.cuda(local_rank) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                state["loss"] = loss
                state["predictions"] = predictions
                state["targets"] = targets
                dataLogger(state)
        
            dataLogger.flush()
            dataLogger.startTest()
            model.eval()
            for batch in dataset.test:
                inputs, targets = (b.cuda(local_rank) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                state["loss"] = loss
                state["predictions"] = predictions
                state["targets"] = targets
                dataLogger(state)

            dataLogger.flush()
            modelSaver(epoch)
