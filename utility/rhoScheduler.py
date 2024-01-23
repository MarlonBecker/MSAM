import math
from utility.args import Args

Args.add_argument("--rhoScheduler", type=str, nargs = "*", help="list of learning rate schedulers")

class _rhoScheduler():
    def __init__(self, optimizer, last_epoch = -1):
        self.optimizer = optimizer
        self.base_rhos = [Args.rho for _ in optimizer.param_groups]
        self.last_epoch = last_epoch

    def _calcFactor(self, progress):
        # self.factor = self.last_epoch * ...
        raise NotImplementedError
        
    def step(self, epoch = None, progress = 0):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        self._calcFactor(progress)

        for group, baseRho in zip(self.optimizer.param_groups, self.base_rhos):
            group['rho'] = baseRho * self.factor

    def get_last_rho(self):
        return [baseRho * self.factor for baseRho in self.base_rhos]

class ConstRho(_rhoScheduler):
    def __init__(self, optimizer, last_epoch = -1):
        super(ConstRho, self).__init__(optimizer, last_epoch = last_epoch)

    def _calcFactor(self, progress):
        self.factor = 1

Args.add_argument("--rhoScheduler_StartRamp_epochs", type=float, help="length of start ramp")
class StartRamp(_rhoScheduler):
    def __init__(self, optimizer, last_epoch = -1):
        super(StartRamp, self).__init__(optimizer, last_epoch = last_epoch)
        
        self.length = Args.rhoScheduler_StartRamp_epochs

    def _calcFactor(self, progress):
        self.factor = min(1, (self.last_epoch + progress)/self.length)

Args.add_argument("--rhoScheduler_StartJump_epochs", type=float, help="length of start ramp")
class StartJump(_rhoScheduler):
    def __init__(self, optimizer, last_epoch = -1):
        super(StartJump, self).__init__(optimizer, last_epoch = last_epoch)
        
        self.length = Args.rhoScheduler_StartJump_epochs

    def _calcFactor(self, progress):
        self.factor = 0 if self.last_epoch + progress <= self.length else 1

class Cosine(_rhoScheduler):
    def __init__(self, optimizer, last_epoch = -1):
        super(Cosine, self).__init__(optimizer, last_epoch = last_epoch)

        self.epochsPerPeriod = Args.epochs * 0.5

    def _calcFactor(self, progress):
        self.factor = 0.5 * (1 + math.cos((self.last_epoch + progress) / self.epochsPerPeriod * 2*math.pi))


class ChainedScheduler(_rhoScheduler):
    def __init__(self, optimizer, schedulerClasses: list, last_epoch = -1):
        super(ChainedScheduler, self).__init__(optimizer, last_epoch = last_epoch)

        self.schedulers = [scheduler(optimizer, last_epoch = last_epoch) for scheduler in schedulerClasses]
        
    def _calcFactor(self, progress):
        self.factor = 1
        for scheduler in self.schedulers:
            scheduler.last_epoch = self.last_epoch
            scheduler._calcFactor(progress)
            self.factor *= scheduler.factor


schedulerDict = {
    "startRamp"  : StartRamp,
    "startJump"  : StartJump,
    "const": ConstRho,
    "cos": Cosine,
}


def getRhoScheduler(optimizer):
    schedulers = []
    for scheduler in Args.rhoScheduler:
        if scheduler in schedulerDict:
            schedulers.append(schedulerDict[scheduler])
        else:
            raise RuntimeError(f"Rho Scheduler {scheduler} not found. Available schedulers: {', '.join(schedulerDict.keys())}")

    if len(schedulers) == 0:
        raise RuntimeError(f"No rho scheduler selected")
    elif len(schedulers) == 1:
        return schedulers[0](optimizer)
    else:
        return ChainedScheduler(optimizer, schedulers)

