import torch.optim as opt

# optimization
LEARNING_RATES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

WEIGHT_DECAY = 1e-4
OPTIMIZATION_METHODS_NAMES = ['adam', 'adamw', 'adagrad', 'rmsprop', 'sgd']


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
    else:
        raise IndexError()
