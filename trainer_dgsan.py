import random

import torch

from loss import DGSANLoss
from trainer import Trainer


class DGSAN(Trainer):
    def __init__(self, writer, device, dataset: str,
                 init_embedd_dim: int, init_hidden_size: int, init_learning_rate: float,
                 dgsan_divergence_type: str,
                 learning_rate_gamma: float, batch_size: int,
                 evaluate_test=True, evaluate_valid=True):
        super(DGSAN, self).__init__(writer=writer, device=device, dataset=dataset, init_embedd_dim=init_embedd_dim,
                                    init_hidden_size=init_hidden_size, init_learning_rate=init_learning_rate,
                                    learning_rate_gamma=learning_rate_gamma, batch_size=batch_size,
                                    evaluate_test=evaluate_test, evaluate_valid=evaluate_valid)

        self.step = 0
        self.token_size = 1

        # model ###################################
        self.old_model = self.make_lstm(embedd_dim=self.embedd_dim, hidden_size=self.hidden_size)
        self.old_model.eval()

        # loss ####################################
        self.dgsan_loss = DGSANLoss(type=dgsan_divergence_type).to(device)

    def dgsan_step(self):
        self.step += 1

        assert self.old_model.embedding_dim == self.model.embedding_dim
        assert self.old_model.hidden_dim == self.model.hidden_dim
        self.old_model.load_state_dict(self.model.state_dict())
        self.old_model.eval()
        torch.cuda.empty_cache()

    def dgsan_step_newsize(self, embedd_dim: int, hidden_size: int):
        self.embedd_dim = embedd_dim
        self.hidden_size = hidden_size

        self.step += 1

        self.old_model = self.make_lstm(clone=self.model)
        self.old_model.eval()

        self.model = self.make_lstm(embedd_dim=embedd_dim, hidden_size=hidden_size)
        self._set_optimizer()

        torch.cuda.empty_cache()

    def write_parameters(self, epoch: int):
        super(DGSAN, self).write_parameters(epoch=epoch)
        self.writer.add_scalar("parameters/dgsan_step", self.step, epoch)
        self.writer.add_scalar("parameters/token_size", self.token_size, epoch)

    def get_random_token_size(self):
        return random.randint(1, self.token_size)
