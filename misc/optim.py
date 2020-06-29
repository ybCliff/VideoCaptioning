import torch.optim as optim

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, learning_rate, minimum_learning_rate, epoch_decay_rate, grad_clip=2):
        self._optimizer = optimizer
        self.n_current_steps = 0
        self.n_current_epochs = 0
        self.lr = learning_rate
        self.mlr = minimum_learning_rate
        self.decay = epoch_decay_rate
        self.grad_clip = grad_clip

    def clip_gradient(self):
        for group in self._optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-self.grad_clip, self.grad_clip)

    def step(self):
        "Step with the inner optimizer"
        self.step_update_learning_rate()
        #self.clip_gradient()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def epoch_update_learning_rate(self):
        self.n_current_epochs += 1
        self.lr = max(self.mlr, self.decay * self.lr)

    def step_update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr

    def get_lr(self):
        return self.lr

def get_optimizer(opt, model):
    optim_mapping = {
        'adam': optim.Adam,
        'rmsprop': optim.RMSprop,
    }

    optim_type = opt['optim'].lower()
    assert optim_type in optim_mapping.keys()

    return ScheduledOptim(
        optimizer=optim_mapping[optim_type](
            filter(lambda p: p.requires_grad, model.parameters()), weight_decay=opt["weight_decay"]),
        learning_rate=opt['alr'],
        minimum_learning_rate=opt['amlr'],
        epoch_decay_rate=opt['decay']
        )