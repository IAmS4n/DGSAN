import numpy as np


class NamedStreamAverage:
    def __init__(self):
        self.means = {}
        self.cnt = {}

    def add(self, name, val, cnt):
        if name not in self.means:
            self.means[name] = 0.
            self.cnt[name] = 0

        self.means[name] = (self.means[name] * self.cnt[name] + val) / (self.cnt[name] + cnt)
        self.cnt[name] += cnt

    def reset(self):
        self.means = {}
        self.cnt = {}

    def __getitem__(self, key):
        return self.means[key]

    def __contains__(self, key):
        return key in self.means

    def __iter__(self):
        return iter(self.means.keys())


class DGSANStep:
    def __init__(self, auto: bool, epoch_num: int = 0, loss_num: int = 20, loss_eps: float = 3e-3,
                 max_epoch: int = 100, min_epoch: int = 10):
        if not auto:
            assert epoch_num > 0

        if auto:
            self.state = -1
        else:
            self.state = epoch_num

        self._last_epoch = 0

        self.loss_num = loss_num
        self.loss_eps = loss_eps
        self.max_epoch = max_epoch
        self.min_epoch = min_epoch
        self.last_losses = [None] * loss_num

        self._last_true_check = 0
        self.cnt = 0

    def check(self, epoch):

        if None in self.last_losses:  # or np.mean(self.last_losses) >= math.log(2.) * 2:
            result = False
        elif epoch <= self._last_true_check + self.min_epoch:
            result = False
        elif epoch >= self._last_true_check + self.max_epoch:
            result = True
        elif self.state > 0:
            last_epoch_mod = self._last_epoch % self.state
            epoch_mod = epoch % self.state
            self._last_epoch = epoch
            result = epoch_mod < last_epoch_mod
        else:
            # result = abs(np.mean(self.last_losses[:-1]) - self.last_losses[-1]) < self.loss_eps
            result = np.std(self.last_losses) < self.loss_eps

        if result:
            self._last_true_check = epoch
            self.cnt += 1
        return result

    def add_loss(self, loss):
        self.last_losses = self.last_losses[1:] + [loss, ]


class BigLRDetector:
    def __init__(self, min_epoch: int = 10, min_diff: int = 20, threshold: float = 0.):

        self.min_epoch = min_epoch
        self.min_diff = min_diff
        self.threshold = threshold

        self.last_loss = None
        self.last_detect = None
        self.ds = []

    def check(self, epoch):
        if (self.last_detect is not None) and epoch - self.last_detect < self.min_diff:
            return False
        if len(self.ds) < self.min_epoch:
            return False

        res = np.mean(np.sign(self.ds)) >= self.threshold
        if res:
            self.last_detect = epoch
            self.ds = []
        return res

    def add_loss(self, loss):
        if self.last_loss is not None:
            self.ds.append(loss - self.last_loss)
        self.last_loss = loss
