import random
import torch
import os

from utility.args import Args

Args.add_argument("--rndSeed", type=bool, help="Whether to set rnd seed.")
Args.add_argument("--deterministic", type=bool, help="Whether use deterministic algorithms.")


def initialize():
    if Args.rndSeed:
        seed = random.getrandbits(32)
    else:
        seed = 42 + torch.distributed.get_rank()

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True

    if Args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False