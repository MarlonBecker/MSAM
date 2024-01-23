import torch
import random

from utility.loss import smooth_crossentropy
from utility.args import Args

Args.add_argument("--ESAMbeta", type=float, help="")
Args.add_argument("--ESAMgamma", type=float, help="")

# see https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
# see https://openreview.net/forum?id=n0OeTdNRG0Q
class ESAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.9,
            weight_decay: float = 1e-2,
            nesterov: bool = False,
            rho: float = 0.3,
            beta: float = 0.5,
            gamma: float = 0.5,
            ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
            beta=beta,
            gamma=gamma,
        )
        super(ESAM, self).__init__(params, defaults)

        self.gamma = gamma

    @torch.no_grad()
    def move_up(self):
        norm = self._grad_norm()
        for group in self.param_groups:
            rho = group['rho']
            scale = rho / (norm + 1e-12)

            for p in group["params"]:
                p.requires_grad = True 
                if p.grad is None:
                    continue
                    
                e_w = p.grad * scale
                
                p.add_(e_w)
                self.state[p]["e_w"] = e_w.detach().clone()

    @torch.no_grad()
    def move_back(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.sub_(self.state[p]["e_w"])

                if random.random() >  group['beta']:
                    p.requires_grad = False


    def step(self, model, inputs, targets):
        # see https://github.com/dydjw9/Efficient_SAM/blob/1e06840c466beaebfafb049096d6a1733cb77a8c/utils/Esam.py

        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True
        
        predictions = model(inputs)

        loss = smooth_crossentropy(predictions, targets)
        loss.mean().backward()
        l_before = loss.clone().detach()

        self.move_up()
        self.zero_grad()

        with torch.no_grad():
            l_after = smooth_crossentropy(model(inputs),targets)
            instance_sharpness = l_after-l_before

            #codes for sorting 
            prob = self.gamma
            if prob >=0.99:
                indices = range(len(targets))
            else:
                position = int(len(targets) * prob)
                cutoff,_ = torch.topk(instance_sharpness,position)
                cutoff = cutoff[-1]
                # cutoff = 0
                #select top k% 
                indices = [instance_sharpness > cutoff] 

        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False
        self.zero_grad()

        # second forward-backward step
        model.module.setBatchNormTracking(False)
        loss_intermediate = smooth_crossentropy(model(inputs[indices]), targets[indices])
        model.module.setBatchNormTracking(True)
        loss_intermediate.mean().backward()

        self.move_back()

        self.SGD_step()
        self.zero_grad()

        return loss, predictions

    def _grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups for p in group["params"] if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    @torch.no_grad()
    def SGD_step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    lr=lr,
                    nesterov=nesterov,
                    )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

def sgd(params,
        d_p_list,
        momentum_buffer_list,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        nesterov: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0: #@TODO decouple weight decay from momentum?
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)

