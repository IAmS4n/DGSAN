import time
from shutil import copyfile
from typing import List

import torch
import torch.optim as optim
from torchtext.data import BucketIterator

from data_manager import load_real_dataset
from metrics.parallel_metrics import MetricsEval
from models import LSTM


class Trainer:
    def __init__(self, writer, device, dataset: str,
                 init_embedd_dim: int, init_hidden_size: int,
                 init_learning_rate: float, learning_rate_gamma: float, batch_size: int,
                 evaluate_test=True, evaluate_valid=True):

        self.writer = writer
        self.device = device
        self.dataset = dataset
        self.embedd_dim = init_embedd_dim
        self.hidden_size = init_hidden_size
        self.learning_rate_gamma = learning_rate_gamma
        self.evaluate_test = evaluate_test
        self.evaluate_valid = evaluate_valid

        # data ####################################
        self._set_data(dataset, batch_size)

        # model ###################################
        self.model = self.make_lstm(embedd_dim=self.embedd_dim, hidden_size=self.hidden_size)

        # optimizer ###############################
        self.init_learning_rate = init_learning_rate
        self.learning_rate = self.init_learning_rate
        self._set_optimizer()

    def _set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.learning_rate_gamma)

    def _set_data(self, dataset: str, batch_size: int):
        train_ds, valid_ds, test_ds, TEXT = load_real_dataset(dataset)

        self._reverse = lambda x: TEXT.reverse(x)

        self.first_token = TEXT.vocab.stoi["<sos>"]
        self.vocab_size = len(TEXT.vocab)
        self.length = TEXT.max_length
        self.train_batchmanager, self.val_batchmanager, self.test_batchmanager = BucketIterator.splits(
            (train_ds, valid_ds, test_ds),
            batch_sizes=(
                batch_size,
                2 * batch_size,
                2 * batch_size),
            device="cpu",
            sort_key=lambda x: len(x.text),
            sort_within_batch=False,
            repeat=False)

        if self.evaluate_test:
            self.eval_test = MetricsEval(test_ds, TEXT, "Test")
        if self.evaluate_valid:
            self.eval_valid = MetricsEval(valid_ds, TEXT, "Valid")

    def pad_random_sequence(self, inp, final_seq_len):
        random_size = final_seq_len - inp.size(1)
        if random_size > 0:
            pad = torch.randint(self.vocab_size, (inp.size(0), random_size)).to(self.device)
            return torch.cat([inp, pad], 1)
        else:
            return inp

    def evaluate(self, samples_text: List[str], loss_log, epoch: int):
        # valid ##############################
        if self.evaluate_valid:
            self.eval_valid.add(epoch=epoch, samples=samples_text)
            self.eval_valid.update_writer(writer=self.writer)

            with torch.no_grad():
                for batch_data in self.val_batchmanager:
                    inp = batch_data.text.to(self.device)
                    nll_loss = -1. * self.model(sentences=inp).mean()
                    loss_log.add("Loss/NLLValid", nll_loss.item() * inp.size(0), inp.size(0))

        # test ###############################
        if self.evaluate_test:
            self.eval_test.add(epoch=epoch, samples=samples_text)
            self.eval_test.update_writer(writer=self.writer)

            with torch.no_grad():
                for batch_data in self.test_batchmanager:
                    inp = batch_data.text.to(self.device)
                    nll_loss = -1. * self.model(sentences=inp).mean()
                    loss_log.add("Loss/NLLTest", nll_loss.item() * inp.size(0), inp.size(0))

    def make_lstm(self, embedd_dim: int = None, hidden_size: int = None, clone=None):
        assert ((embedd_dim is not None) and (hidden_size is not None)) or (clone is not None)
        if clone is not None:
            model = LSTM(clone.embedding_dim, clone.hidden_dim,
                         vocab_size=self.vocab_size,
                         tagset_size=self.vocab_size)
            model.load_state_dict(clone.state_dict())
        else:
            model = LSTM(embedd_dim, hidden_size, vocab_size=self.vocab_size, tagset_size=self.vocab_size)
        model = model.to(self.device)
        return model

    def get_samples_text(self):
        with torch.no_grad():
            # NOTE : changing number of sample effects on metrics such as MS-jaccard
            samples = self.model.sample(number=1024, seq_len=self.length, first_token=self.first_token,
                                        device=self.device).cpu()
        samples_text = self._reverse(samples)
        return samples_text

    def dec_lr(self):
        self.scheduler.step()
        self.learning_rate = self.scheduler.get_lr()[0]

    def rst_lr(self):
        self.learning_rate = self.init_learning_rate
        self._set_optimizer()

    def write_parameters(self, epoch: int):
        self.writer.add_scalar("parameters/embedd_dim", self.embedd_dim, epoch)
        self.writer.add_scalar("parameters/hidden_size", self.hidden_size, epoch)
        self.writer.add_scalar("parameters/learning_rate", self.learning_rate, epoch)

    def evaluation_phase(self, epoch, loss_log, back_save_path, samples_path):
        torch.save(self.model.state_dict(), back_save_path + "_%04d.pth" % epoch)

        samples_text = self.get_samples_text()

        with open(samples_path + "_%04d.txt" % epoch, "w") as sample_file:
            sample_file.write("\n".join(samples_text))
        self.evaluate(samples_text, loss_log, epoch=epoch)

        for loss_name in loss_log:
            self.writer.add_scalar(loss_name, loss_log[loss_name], epoch)
        self.writer.add_text("samples", "\n\n".join(samples_text[:32]), epoch)

    def finish_evaluation(self):
        unfinished = True
        while unfinished:
            unfinished = False

            if self.evaluate_valid:
                if (not self.eval_valid.to_ngram_process.empty()) or (
                        not self.eval_valid.to_bert_process.empty()):
                    self.eval_valid.update_writer(writer=self.writer, block=True)
                    unfinished = True

            if self.evaluate_test:
                if (not self.eval_test.to_ngram_process.empty()) or (not self.eval_test.to_bert_process.empty()):
                    self.eval_test.update_writer(writer=self.writer, block=True)
                    unfinished = True
            time.sleep(5 * 60)
        time.sleep(10 * 60)

        if self.evaluate_valid:
            self.eval_valid.update_writer(writer=self.writer)
            self.eval_valid.print_status()
        if self.evaluate_test:
            self.eval_test.update_writer(writer=self.writer)
            self.eval_test.print_status()

    def _config_str(self):
        return "%s__%d__%d__%d__%d__%d" % (
            self.dataset, self.vocab_size, self.first_token, self.length, self.embedd_dim, self.hidden_size)

    def save_last(self, best_save_path):
        final_path = best_save_path + "___last___%s.pth" % self._config_str()
        torch.save(self.model.state_dict(), final_path)

    def select_best_save(self, back_save_path, best_save_path):
        for eval_res_name in self.eval_valid.bests:
            best_epoch, best_score = self.eval_valid.bests[eval_res_name]
            print(eval_res_name, "Best score:", best_score, "Best Epoch", best_epoch)
            best_path = back_save_path + "_%04d.pth" % best_epoch

            metric_simply_name = eval_res_name.replace("/", "_").replace("\\", "_").replace(".", "_")
            final_path = best_save_path + "___best_%s_%04d___%s.pth" % (metric_simply_name, best_epoch, self._config_str())
            copyfile(best_path, final_path)
