import torch

from .sgd import SGD
from .sam import SAM
from .msam import MSAM
from .esam import ESAM
from .adamW import AdamW
from .adamW_msam import AdamW_MSAM
from .adamW_sam import AdamW_SAM
from .looksam import LookSAM
from utility.args import Args

Args.add_argument("--optimizer", type=str, help="optimizer name")
Args.add_argument("--weightDecay", type=float, help="L2 weight decay.")
Args.add_argument("--momentum", type=float, help="Momentum.")
Args.add_argument("--rho", type=float, help="")
Args.add_argument("--nesterov", type=bool, help="use normal nesterov momentum for sgd/sam")

Args.add_argument("--grad_clip_norm", type=float, help="")


def getOptimizer(params) -> torch.nn.Module:
    # optimizer names and additional args which will be passed from Args.XX
    optimizerDict = {
        "SGD":        (SGD,        {"momentum": Args.momentum, "nesterov": Args.nesterov}),
        "SAM":        (SAM,        {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": Args.rho}),
        "ESAM":       (ESAM,       {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": Args.rho}),
        "lookSAM":    (LookSAM,    {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": Args.rho}),
        "MSAM":       (MSAM,       {"momentum": Args.momentum, "rho": Args.rho}),
        "AdamW":      (AdamW,      {}),
        "AdamW_MSAM": (AdamW_MSAM, {"rho": Args.rho}),
        "AdamW_SAM":  (AdamW_SAM,  {"rho": Args.rho}),
    }
    
    if Args.optimizer in optimizerDict:
        optimizer, additionalArgs = optimizerDict[Args.optimizer]
        return optimizer(params, lr = Args.learningRate, weight_decay=Args.weightDecay, **additionalArgs)
    else:
        raise RuntimeError(f"Optimizer '{Args.optimizer}' not found. Available optimizers: {', '.join(optimizerDict.keys())}")
