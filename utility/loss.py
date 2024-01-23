import torch
import torch.nn.functional as F

from utility.args import Args

Args.add_argument("--label_smoothing", type=float, help="Smoothing for smooth_crossentropy")

def smooth_crossentropy(pred, targets):
    smoothing = Args.label_smoothing
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=targets.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
