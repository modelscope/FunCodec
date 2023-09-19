import torch
import torch.optim._functional as F
from torch.optim.optimizer import Optimizer


class LazyAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=False, maximize=False)
        super(LazyAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LazyAdamW, self).__setstate__(state)

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

        for group in self.param_groups:
            dense_params_with_grad, sparse_params_with_grad = [], []
            dense_grads, sparse_grads = [], []
            dense_exp_avgs, sparse_exp_avgs = [], []
            dense_exp_avg_sqs, sparse_exp_avg_sqs = [], []
            dense_state_steps, sparse_state_steps = [], []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    sparse_params_with_grad.append(p)
                    sparse_grads.append(p.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = torch.zeros([], dtype=torch.int64)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    sparse_exp_avgs.append(state['exp_avg'])
                    sparse_exp_avg_sqs.append(state['exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    sparse_state_steps.append(state['step'])

                else:
                    dense_params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    dense_grads.append(p.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = torch.zeros([], dtype=torch.int64)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    dense_exp_avgs.append(state['exp_avg'])
                    dense_exp_avg_sqs.append(state['exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    dense_state_steps.append(state['step'])

            if len(dense_params_with_grad) > 0:
                F.adamw(
                    dense_params_with_grad,
                    dense_grads,
                    dense_exp_avgs,
                    dense_exp_avg_sqs,
                    [],
                    dense_state_steps,
                    amsgrad=False,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=False
                )

            if len(sparse_params_with_grad) > 0:
                F.sparse_adam(
                    sparse_params_with_grad,
                    sparse_grads,
                    sparse_exp_avgs,
                    sparse_exp_avg_sqs,
                    sparse_state_steps,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    eps=group['eps']
                )

        return loss