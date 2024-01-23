from typing import List
from torch import Tensor
import torch
import math

# cf. https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
class AdamW_MSAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            maximize: bool = False,
            rho = 1,
            ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            rho=rho,
        )
        super(AdamW_MSAM, self).__init__(params, defaults)

        for group in self.param_groups:
            group["norm_factor"] = [0,]

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if len(self.param_groups) > 1:
            raise RuntimeError("only one parameter group supported atm for MSAM")
        group = self.param_groups[0]

        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_sums = []
        state_steps = []
        beta1, beta2 = group['betas']

        for p in group['params']:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError('AdamW does not support sparse gradients')
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])


            # update the steps for each param group update
            state['step'] += 1
            # record the step after step update
            state_steps.append(state['step'])

        adamW_msam(params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                rho=group['rho'],
                norm_factor=group['norm_factor'],
                )

        return loss


    @torch.no_grad()
    def move_up_to_momentumAscent(self):
        for group in self.param_groups:
            for p in group['params']:
                if "exp_avg" in self.state[p]:
                    p.sub_(self.state[p]["exp_avg"], alpha = group["norm_factor"][0])

    @torch.no_grad()
    def move_back_from_momentumAscent(self):
        for group in self.param_groups:
            for p in group['params']:
                if "exp_avg" in self.state[p]:
                    p.add_(self.state[p]["exp_avg"], alpha = group["norm_factor"][0])


def adamW_msam(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          state_steps: List[int],
          *,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float,
          rho:float,
          norm_factor: list,
          ):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # descent
        param.add_(exp_avg, alpha = norm_factor[0])

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)

        param.mul_(1 - lr * weight_decay)


    #calculate ascent step norm
    ascent_norm = torch.norm(
                torch.stack([
                    buf.norm(p=2)
                    for buf in exp_avgs
                ]),
                p=2
        )
    norm_factor[0] = 1/(ascent_norm+1e-12) * rho
    
    #ascent
    for i, param in enumerate(params):
        param.sub_(exp_avgs[i], alpha = norm_factor[0])

