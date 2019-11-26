import torch.optim as opt

# optimization
LEARNING_RATES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

WEIGHT_DECAY = 1e-4
OPTIMIZATION_METHODS_NAMES = ['adam', 'adamw', 'adagrad', 'rmsprop', 'sgd', 'radam']

import torch
from torch.optim.optimizer import Optimizer, required
import math

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

def radam(model, lr, **kwargs):
    return RAdam(model.parameters(), lr)

def adamw(model, lr, **kwargs):
    return opt.AdamW(model.parameters(), lr)


def adam(model, lr, **kwargs):
    # learning_rate_tensor = tf.Variable(initial_value=learning_rate, trainable=True)
    # logging
    # with tf.name_scope('adam'):
    #     tf.summary.scalar('learning_rate', learning_rate_tensor)

    return opt.Adam(model.parameters(), lr)


def adagrad(model, lr, **kwargs):
    return opt.Adagrad(model.parameters(), lr)


def rmsprop(model, lr, **kwargs):
    return opt.RMSprop(model.parameters(), lr)


def sgd(model, lr, **kwargs):
    return opt.SGD(model.parameters(), lr)


def optimizer(optimizer_iteration, learning_rate_iteration, parametrization, task_metadata):
    if optimizer_iteration == 1:
        return adamw(model=parametrization, lr=LEARNING_RATES[learning_rate_iteration], weight_decay=WEIGHT_DECAY)
    elif optimizer_iteration == 0:
        return adam(model=parametrization, lr=LEARNING_RATES[learning_rate_iteration])
    elif optimizer_iteration == 2:
        return adagrad(model=parametrization, lr=LEARNING_RATES[learning_rate_iteration])
    elif optimizer_iteration == 3:
        return rmsprop(model=parametrization, lr=LEARNING_RATES[learning_rate_iteration])
    elif optimizer_iteration == 5:
        return radam(model=parametrization, lr=LEARNING_RATES[learning_rate_iteration])
    else:
        raise IndexError()
