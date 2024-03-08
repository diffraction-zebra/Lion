from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach)


class Lion(Optimizer):
    def __init__(self, params,
                 lr=1e-4,
                 betas=(0.9, 0.99),
                 weight_decay=0.1,
                 *,
                 maximize: bool = False,
                 foreach: Optional[bool] = None,
                 differentiable: bool = False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas value: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr,
                        betas=betas,
                        weight_decay=weight_decay,
                        maximize=maximize,
                        foreach=foreach,
                        differentiable=differentiable)

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    # accumulate updated parameters, grads, moments
    def _init_group(self, group):
        params_with_grad = []
        grads = []
        moments = []

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)
                # there was checking for sparse but I move it to algo implementation

                state = self.state[p]
                if 'momentum' not in state:
                    moments.append(None)
                else:
                    moments.append(state['momentum'])

        return params_with_grad, grads, moments

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            params_with_grad, grads, moments = self._init_group(group)

            lion(params_with_grad,
                 grads,
                 moments,
                 weight_decay=group['weight_decay'],
                 lr=group['lr'],
                 betas=group['betas'],
                 maximize=group['maximize'],
                 foreach=group['foreach'])

            # update momentum for trainable params
            for p, momentum in zip(params_with_grad, moments):
                state = self.state[p]
                state['momentum'] = momentum

        return loss


def lion(params: List[Tensor],
         grads: List[Tensor],
         moments: List[Optional[Tensor]],
         foreach: Optional[bool] = None,
         *,
         weight_decay: float,
         lr: float,
         betas: Tuple[float, float],
         maximize: bool):
    if foreach is None:
        # JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
        else:
            foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_lion
    else:
        func = _single_tensor_lion

    func(params,
         grads=grads,
         moments=moments,
         weight_decay=weight_decay,
         lr=lr,
         betas=betas,
         maximize=maximize)


def _single_tensor_lion(params: List[Tensor],
                        grads: List[Tensor],
                        moments: List[Optional[Tensor]],
                        *,
                        weight_decay: float,
                        lr: float,
                        betas: Tuple[float, float],
                        maximize: bool):
    b1, b2 = betas
    # use inplace operations where it's possible
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        # copying is unavoidable
        update = grad.clone().detach()

        momentum = moments[i]

        if momentum is None:
            # copy for future updates
            moments[i] = torch.clone(grad).detach()
        else:
            # add momentum to update
            update.mul_(1 - b1).add_(momentum, alpha=b1)
            # carrying about the moment
            momentum.mul_(b2).add_(grad, alpha=1 - b2)

        if update.is_sparse and not update.is_coalesced():
            # sign_ operation currently don't support uncoalesce tensors
            update = update.coalesce()
        update.sign_()

        if weight_decay != 0:
            # adding to update tensor unable in case of sparsity grad:
            # add(sparse, dense) is not supported
            param.mul_(1 - lr * weight_decay)

        param.add_(update, alpha=-lr)

        del update


def _group_tensors_by_sparse_grads(params: List[Tensor],
                                   grads: List[Tensor],
                                   moments: List[Tensor]):
    dense_p, sparse_p = list(), list()
    dense_g, sparse_g = list(), list()
    dense_m, sparse_m = list(), list()
    dense_idx, sparse_idx = list(), list()

    for i, (p, g, m) in enumerate(zip(params, grads, moments)):
        if g.is_sparse:
            sparse_p.append(p)
            sparse_g.append(g)
            sparse_m.append(m)
            sparse_idx.append(i)
        else:
            dense_p.append(p)
            dense_g.append(g)
            dense_m.append(m)
            dense_idx.append(i)

    return ([dense_p, dense_g, dense_m, dense_idx],
            [sparse_p, sparse_g, sparse_m, sparse_idx])


def _multi_tensor_lion(params: List[Tensor],
                       grads: List[Tensor],
                       moments: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       lr: float,
                       betas: Tuple[float, float],
                       maximize: bool):
    if len(params) == 0:
        return
    dense, sparse = _group_tensors_by_sparse_grads(params, grads, moments)

    # use simple for-loop implementation for params with sparse gradients
    _single_tensor_lion(sparse[0],
                        sparse[1],
                        sparse[2],
                        weight_decay=weight_decay,
                        lr=lr,
                        betas=betas,
                        maximize=maximize)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(dense[:3], with_indices=True)
    dense_idx = dense[3]
    for ((device_params, device_grads, device_moments), indices) in grouped_tensors.values():

        # use foreach realisation for parameters with dense gradients
        b1, b2 = betas
        if maximize:
            device_grads = torch._foreach_neg_(device_grads)

        updates = [grad.clone().detach() for grad in device_grads]

        all_states_with_moment = True
        for m in device_moments:
            if m is None:
                all_states_with_moment = False
                break

        if all_states_with_moment:
            updates_with_mom, grads_with_mom, moments_with_mom = updates, device_grads, device_moments
        else:
            updates_with_mom, moments_with_mom, grads_with_mom = list(), list(), list()

            for i, (u, m, g) in enumerate(zip(updates, device_moments, device_grads)):
                if m is not None:
                    updates_with_mom.append(u)
                    moments_with_mom.append(m)
                    grads_with_mom.append(g)
                else:
                    moments[dense_idx[indices[i]]] = device_grads[i].clone().detach()

        if len(updates_with_mom) > 0:
            torch._foreach_mul_(updates_with_mom, 1 - b1)
            torch._foreach_add_(updates_with_mom, moments_with_mom, alpha=b1)

        torch._foreach_sign_(updates)

        if len(updates_with_mom) > 0:
            torch._foreach_mul_(moments_with_mom, b2)
            torch._foreach_add_(moments_with_mom, grads_with_mom, alpha=1 - b2)

        if weight_decay != 0:
            torch._foreach_add_(updates, device_params, alpha=weight_decay)
        torch._foreach_add_(device_params, updates, alpha=-lr)

        for u in updates:
            del u
