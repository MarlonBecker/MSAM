import h5py
import time
import torch
import os

from utility.args import Args
from utility.metrics import BaseMetric, available_metrics

Args.add_argument("--truncate", type=bool, help="truncate log file")
Args.add_argument("--verbose", type=bool, help="print to terminal")
Args.add_argument("--metrics", type=str, nargs = "*", help="list of metrics")
Args.add_argument("--logEach", type=int, help="Iterations to log during training.")

class DataLogger:
    available_metrics = {}
    def __init__(self):
        self.verbose = Args.verbose
        self.logEach = Args.logEach

        self.columnLen = 15
        self.state = {}
        self.step = 0
        self.epoch = 0
        self.train = True
        self.trainDataLen = None

        self.metrics: list[BaseMetric] = []
        self.printTrainMetrics: list[BaseMetric] = []
        self.printTestMetrics: list[BaseMetric] = []
        for metricName in Args.metrics:
            if metricName in available_metrics:
                self.metrics.append(available_metrics[metricName]())
                if self.metrics[-1].printTrain:
                    self.printTrainMetrics.append(self.metrics[-1])
                if self.metrics[-1].logTest:
                    self.printTestMetrics.append(self.metrics[-1])
            else:
                raise RuntimeError(f"Metric {metricName} not found. Available metrics: {', '.join(available_metrics.keys())}")

        self.loading_bar = LoadingBar(length=(self.columnLen+1)*len(self.printTestMetrics)-3)

        self.filePath = os.path.join(Args.logDir, Args.logSubDir, "data.hdf5")
        if torch.distributed.get_rank() == 0:
            if not Args.contin:
                try:
                    with h5py.File(self.filePath, "w" if Args.truncate else "w-") as f:
                        f.create_group("train")
                        f.create_group("test")
                        
                        for metric in self.metrics:
                            metric.createDatasets(f)
                except FileExistsError as e:
                    raise FileExistsError("To overwrite existing logfile use '--truncate' option. ") from e
            else:
                if not os.path.isfile(self.filePath):
                    raise RuntimeError(f"Running in continue mode but log file not found. Path: {self.filePath}")


    def __call__(self, state: dict) -> None:
        for metric in self.metrics:
            if self.train and metric.logTrain:
                metric.fetchMetric(state)
            if not self.train and metric.logTest:
                metric.fetchMetric(state)

        self.step += 1
        if torch.distributed.get_rank() == 0:
            if self.verbose and self.step % self.logEach == self.logEach - 1:
                self.printTerminal()

    def startTrain(self, trainDataLen) -> None:
        self.epoch += 1
        self.trainDataLen = trainDataLen
        self.step = 0
        self.train = True
        self.start_time = time.time()
        

    def startTest(self) -> None:
        self.step = 0
        self.train = False

    def flush(self) -> None:
        if torch.distributed.get_rank() == 0:
            self.printTerminal()
            if not self.train:
                print()
            with h5py.File(self.filePath, "r+") as file:
                for metric in self.metrics:
                    if self.train and metric.logTrain:
                        metric.flushData(file, mode = "train" if self.train else "test")
                    if not self.train and metric.logTest:
                        metric.flushData(file, mode = "train" if self.train else "test")
            

    def printTerminal(self) -> None:
        """ print to terminal """
        if self.train:
            self.trainString = f"{'│'.join([metric.getDisplayStr().center(self.columnLen) for metric in self.printTrainMetrics])}"
            
            if self.verbose:
                print(f"\r┃{str(self.epoch).center(self.columnLen-1)}│{self._time().center(self.columnLen-1)}┃{self.trainString}{self.loading_bar(self.step / self.trainDataLen)}",
                    end="",
                    flush=True)
        else:
            start = '\r' if self.verbose else ''
            print(f"{start}┃{str(self.epoch).center(self.columnLen-1)}│{self._time().center(self.columnLen-1)}┃{self.trainString}┃{'│'.join([metric.getDisplayStr().center(self.columnLen) for metric in self.printTestMetrics])}┃",
                end="")

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def printHeader(self) -> None:
        if torch.distributed.get_rank() == 0:
            print(f"┏━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳{'T╺╸R╺╸A╺╸I╺╸N '.center((self.columnLen+1)*len(self.printTrainMetrics)-1,'━')}┳{'T╺╸E╺╸S╺╸T '.center((self.columnLen+1)*len(self.printTestMetrics)-1,'━')}┓")
            print(f"┃                             ┃{' '*((self.columnLen+1)*len(self.printTrainMetrics)-1)}┃{' '*((self.columnLen+1)*len(self.printTestMetrics)-1)}┃")
            print(f"┃    epoch     │     time     ┃{'│'.join([metric.name[:self.columnLen].center(self.columnLen) for metric in self.printTrainMetrics])}┃{'│'.join([metric.name[:self.columnLen].center(self.columnLen) for metric in self.printTestMetrics])}┃")
            print(f"┠──────────────┼──────────────╂{'┼'.join(['─'*self.columnLen]*len(self.printTrainMetrics))}╂{'┼'.join(['─'*self.columnLen]*len(self.printTestMetrics))}┨")

class LoadingBar:
    def __init__(self, length: int = 40):
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"
