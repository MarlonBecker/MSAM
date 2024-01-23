import torch
import h5py
import numpy as np

class BaseMetric:
    name: str = None
    logTrain: bool = True
    logTest: bool = True
    printTrain: bool = True
    printTest: bool = True
    maxIterations: int = None
    shape: tuple[int] = () #defines shape of data (excluding length/batch/epoch dimension)
    consolePrintFormat: str = ".4f"
    concatenateWorkers: bool = True #whether to concatenate data from all workers after iteration or just use local_rank==0's data
    def __init__(self):
        self.buffer = []
        if not self.logTest:
            self.printTest = False
        if not self.logTrain:
            self.printTrain = False

    def getDisplayStr(self) -> float:
        """reduce buffer and generate str for command line output"""
        return f"{np.mean(np.concatenate(self.buffer)):{self.consolePrintFormat}}"

    def _reduceData(self):
        """reduce data to write to buffer
        default: no reduction
        
        return np.array or convertible to np.array which has shape [dataLen, self.shape], where dataLen is concatenated into file
        return values has to have at least 1din (i.e. len(...)==1)
        """
        return [np.mean(np.concatenate(self.buffer))]


    def fetchMetric(self, state: dict):
        """
        calls self.calcMetric
        gather data from workers
        """
        worker_metric: np.ndarray = self.calcMetric(state)

        if self.concatenateWorkers:
            gather_list = [None] * torch.distributed.get_world_size()
            torch.distributed.gather_object(
                                    worker_metric,
                                    object_gather_list = gather_list if torch.distributed.get_rank() == 0 else None,
                                    dst = 0
                                    )
            
            if torch.distributed.get_rank() == 0:
                self.buffer.append(np.concatenate(gather_list, axis = 0))
        else:
            if torch.distributed.get_rank() == 0:
                self.buffer.append(worker_metric)
            

                    
    def calcMetric(self, state: dict):
        """
        calcMetric should calculate metric from 'state', move it to cpu, convert to np.ndarray, and make it ready for reduction by _reduceData
        return: np.array
        """
        raise NotImplementedError


    def createDatasets(self, file: h5py.File):
        if self.logTrain:
            file["train"].create_dataset(self.name, shape=(0, *self.shape), dtype=float, maxshape=(self.maxIterations, *self.shape), chunks=True)
        if  self.logTest:
            file["test"].create_dataset(self.name, shape=(0, *self.shape), dtype=float, maxshape=(self.maxIterations, *self.shape), chunks=True)

    def flushData(self, file: h5py.File, mode: str):

        if len(self.buffer) == 0:
            raise RuntimeError(f"No data collected for metric {self.name} in set {mode}")

        data = self._reduceData()

        dataset = file[mode][self.name]
        dataset.resize(len(dataset)+len(data), axis = 0)
        dataset[len(dataset)-len(data):, ...] = np.array(data)

        self.buffer = []



available_metrics = {}
def addMetric(class_):
    if class_.name is None:
        raise ValueError(f"Metric.name has to be defined in metric class definition for metric {class_}.")
    available_metrics[class_.name] = class_
    return class_


"""
 - to create new metrics, simply define classes here which inherit from BaseMetric and decorate with @addMetric
 - overwrite calcMetric and other needed functions (see definitions above)
"""

@addMetric
class MetricLoss(BaseMetric):
    name = "loss"
    def calcMetric(self, state: dict):
        return state["loss"].cpu().numpy()

@addMetric
class MetricLossPerSample(BaseMetric):
    name = "lossPS"
    def calcMetric(self, state: dict):
        return state["loss"].cpu().numpy()
    def _reduceData(self):
        return np.concatenate(self.buffer)

@addMetric
class MetricLR(BaseMetric):
    logTest: bool = False
    concatenateWorkers = False
    name = "learningRate"
    consolePrintFormat = ".3e"
    def calcMetric(self, state: dict):
        return np.array(state["lrScheduler"].get_last_lr())

@addMetric
class MetricAccuracy(BaseMetric):
    name = "accuracy"
    consolePrintFormat = ".2%"
    def calcMetric(self, state: dict):
        return (torch.argmax(state["predictions"].data, 1) == state["targets"]).to(float).cpu().numpy()
