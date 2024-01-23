import torch
from utility.args import Args
from utility.loss import smooth_crossentropy

Args.add_argument("--lookSAMk", type=int, help="")
Args.add_argument("--lookSAMalpha", type=float, help="")


# see https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py
# see https://arxiv.org/abs/2203.02714
class LookSAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.9,
            weight_decay: float = 1e-2,
            nesterov: bool = False,
            rho: float = 0.3,
            alpha: float = 0.2,
            k: int = 5,
            ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
            alpha=alpha,
            k=k,
        )
        super(LookSAM, self).__init__(params, defaults)

        # init momentum buffer to zeros
        # needed to make implementation of first ascent step cleaner (before SGD.step() was ever called) 
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state["momentum_buffer"] = torch.zeros_like(p).detach()

        for group in self.param_groups:
            group["inverse_norm_buffer"] = [0,]

        self.k = k
        self.iteration_counter = k

    @torch.no_grad()
    def move_up(self):
        norm = self._grad_norm()
            
        for group in self.param_groups:
            rho = group['rho']
            scale = rho / (norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                e_w = p.grad * scale
                
                p.add_(e_w)
                self.state[p]["e_w"] = e_w.detach().clone()
                self.state[p]["grad"] = p.grad.detach().clone()

    @torch.no_grad()
    def move_back(self):
        scalar_product = self._scalar_product()
        grad_norm = self._calc_norm(get_var_from_p = lambda x: self.state[x]["grad"])
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.sub_(self.state[p]["e_w"])
                
                #not normalized yet! (see next loop)
                self.state[p]["normalized_orthogonal_gradient"] = (p.grad - scalar_product.to(p)/(grad_norm.to(p)**2+1e-12) * self.state[p]["grad"]).detach().clone()
                
        #normalize orthogonal gradient
        grad_v_norm = self._calc_norm(get_var_from_p = lambda x: self.state[x]["normalized_orthogonal_gradient"])
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["normalized_orthogonal_gradient"].div_(grad_v_norm.to(p))

    @torch.no_grad()
    def update_gradient(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.grad.add_(self.state[p]["normalized_orthogonal_gradient"], alpha = group['alpha'] * grad_norm.to(p))

    def step(self, model, inputs, targets):
        predictions = model(inputs)

        loss = smooth_crossentropy(predictions, targets)
        loss.mean().backward()

        if self.iteration_counter == self.k:
            self.iteration_counter = 0
            
            self.move_up()
            self.zero_grad()

            # second forward-backward step
            model.module.setBatchNormTracking(False)
            loss_intermediate = smooth_crossentropy(model(inputs), targets)
            model.module.setBatchNormTracking(True)
            loss_intermediate.mean().backward()

            self.move_back()
        else:
            self.update_gradient()
            

        self.SGD_step()
        self.zero_grad()

        self.iteration_counter += 1

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
    
    def _calc_norm(self, get_var_from_p):
        """generic norm calculator (could be use form _grad_norm too)

        Args:
            get_var_from_p (callabe): getter for variable to compute norm of
        """
        norm = torch.norm(
                    torch.stack([
                        get_var_from_p(p).norm(p=2) for group in self.param_groups for p in group["params"] if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def _scalar_product(self):
        dot_prod = torch.sum(
                    torch.stack([
                        (p.grad*self.state[p]["grad"]).sum() for group in self.param_groups for p in group["params"] if p.grad is not None
                    ]),
               )
        return dot_prod



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

