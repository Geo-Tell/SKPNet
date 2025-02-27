# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

class AverageMeter_list(object):
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        if type(values).__name__ == 'list':
            for idx, v in enumerate(values):
                self._val[idx] = v
                self._sum[idx] += v
                self._count[idx] += 1
        else:
            self._val[0] = values
            self._sum[0] += values
            self._count[0] += 1

    def val(self, idx=None):
        if idx is None:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]
        else:
            return self._val[idx]

    def count(self, idx=None):
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None):
        if idx is None:
            return self._sum[0] / self._count[0] if self.items is None else [
                self._sum[i] / self._count[i] for i in range(self.n_items)
            ]
        else:
            return self._sum[idx] / self._count[idx]


class AverageMeter(object):
    '''
    Computes ans stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val  # current value
        if not isinstance(val, list):
            self.sum += val  # accumulated sum, n = batch_size
            self.count += 1  # accumulated count
        else:
            self.sum += sum(val)
            self.count += len(val)
        self.avg = self.sum / self.count  # current average value


class LossRecorder(object):
    def __init__(self):
        '''
        Log loss data
        :param config: configuration file.
        :param phase: train, validation or test.
        '''
        self._loss_recorder = {}


    @property
    def loss_recorder(self):
        return self._loss_recorder

    @property
    def loss_recorder_avg(self):
        avg_dict = dict()
        for key in self._loss_recorder:
            avg_dict[key] = self._loss_recorder[key].avg
        return avg_dict

    def update_loss(self, loss_dict):
        for key, item in loss_dict.items():
            if key not in self._loss_recorder:
                self._loss_recorder[key] = AverageMeter()
            self._loss_recorder[key].update(item)


class LogBoard(object):
    def __init__(self, pth):
        self.writer = SummaryWriter(pth)
        self.iter = 1

    def update(self, value_dict, step_len, phase, per_epoch=False):
        n_iter = self.iter * step_len
        extra_tag = 'Epoch' if per_epoch else 'Batch'

        for key, item in value_dict.items():
            self.writer.add_scalar(key + '/' + phase + '/' + extra_tag, item, n_iter)
        self.iter += 1
