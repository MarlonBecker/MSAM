import torch

# see https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
class MSAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.9,
            weight_decay: float = 1e-2,
            nesterov: bool = False,
            rho: float = 0.3,
            ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
        )
        super(MSAM, self).__init__(params, defaults)

        # init momentum buffer to zeros
        # needed to make implementation of first ascent step cleaner (before SGD.step() was ever called) 
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state["momentum_buffer"] = torch.zeros_like(p).detach()

        for group in self.param_groups:
            group["inverse_norm_buffer"] = 0

    @torch.no_grad()
    def step(self):
        # see https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
        if len(self.param_groups) > 1:
            raise RuntimeError("only one parameter group supported atm for MSAM")
        group = self.param_groups[0]
        params_with_grad = []
        d_p_list = []
        momentum_buffer_list = []
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        rho = group['rho']
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

        #### see torch functional sgd
        ## first loop: 
        # p_{t+1}' = p_t + rho*v_t/||v_t|| <- normal p (no descent)
        # v_{t+1} = mu * v_t + g
        # p_{t+1}'' = p_{t+1}' + -lr v_{t+1} <- SGD step
        for i, param in enumerate(params_with_grad):
            d_p = d_p_list[i]
            if weight_decay != 0: 
                d_p.add_(param, alpha=weight_decay) # warning! inplace operation on gradient here

            buf = momentum_buffer_list[i]

            param.add_(buf, alpha = rho*group["inverse_norm_buffer"])
            
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p)
                
            param.add_(buf, alpha = -lr)

        #calculate ascent step norm
        ascent_norm = torch.norm(
                    torch.stack([
                        buf.norm(p=2) for buf in momentum_buffer_list
                    ]),
                    p=2
            )
        group["inverse_norm_buffer"] = 1/(ascent_norm+1e-12)

        # second loop: 
        # p_{t+1} = p_{t+1}'' - rho*v_{t+1}/||v_{t+1}|| <- ascending again
        for i, param in enumerate(params_with_grad):
            param.sub_(momentum_buffer_list[i], alpha = rho*group["inverse_norm_buffer"])
            # slightly more efficient one step update
            # param.sub_(buf, alpha = rho*(-old_norm/momentum + group["inverse_norm_buffer"])+lr).sub_(d_p_list[i], alpha = rho*old_norm/momentum)
            
        # update momentum_buffers in state
        for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            state = self.state[p]
            state['momentum_buffer'] = momentum_buffer


    @torch.no_grad()
    def move_up_to_momentumAscent(self):
        for group in self.param_groups:
            for p in group['params']:
                if "momentum_buffer" in self.state[p]:
                    p.sub_(self.state[p]["momentum_buffer"], alpha = group["rho"]*group["inverse_norm_buffer"])

    @torch.no_grad()
    def move_back_from_momentumAscent(self):
        for group in self.param_groups:
            for p in group['params']:
                if "momentum_buffer" in self.state[p]:
                    p.add_(self.state[p]["momentum_buffer"], alpha = group["rho"]*group["inverse_norm_buffer"])
