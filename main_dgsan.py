import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainer_dgsan import DGSAN
from utils import NamedStreamAverage, DGSANStep, BigLRDetector

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.autograd.set_detect_anomaly(True)


parser = argparse.ArgumentParser(prog='DGSAN')
parser.add_argument('--batch_size', type=int, required=False, default=128)
parser.add_argument('--dataset', type=str, required=False, default="")
parser.add_argument('--details', type=str, required=False, default="")

parser.add_argument('--epoch', type=int, required=False, default=1000)

parser.add_argument('--log_path', type=str, required=False, default="./log")
parser.add_argument('--samples_path', type=str, required=False, default="./samples")
parser.add_argument('--save_path', type=str, required=False, default="./save")
parser.add_argument('--back_save_path', type=str, required=False, default="./back_save")
parser.add_argument('--load', action='store_true')

parser.add_argument('--rnn_embedd_dim', type=int, required=False, default=128)
parser.add_argument('--rnn_hidden_size', type=int, required=False, default=64)

parser.add_argument('--learning_rate', type=float, required=False, default=1e-3)
parser.add_argument('--learning_rate_gamma', type=float, required=False, default=0.75)

parser.add_argument('--loss_eps', type=float, required=False, default=3e-3)
parser.add_argument('--step_max_epoch', type=int, required=False, default=100)
parser.add_argument('--step_min_epoch', type=int, required=False, default=10)

parser.add_argument('--dgsan_divergence_type', type=str, required=False, default="JS")

args = parser.parse_args()

os.makedirs(args.log_path, exist_ok=True)
os.makedirs(args.samples_path, exist_ok=True)
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(args.back_save_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device', device)

# log ########################################################################################
data_abs = args.dataset
net_abs = "RNN%d-%d" % (args.rnn_embedd_dim, args.rnn_hidden_size)
train_abs = args.dgsan_divergence_type
big_lr_detector = BigLRDetector()

time_str = datetime.now().strftime("%y%m%d_%H%M%S").replace("/", "-")
run_name = "%s_Data%s_Net%s_%s" % (time_str, data_abs, net_abs, train_abs)

log_path = os.path.join(args.log_path, run_name)
writer = SummaryWriter(log_path)
if len(args.details) > 0:
    writer.add_text("Details", args.details)
writer.add_text("Arg", str(args))

samples_path = os.path.join(args.samples_path, run_name)
save_path = os.path.join(args.save_path, run_name)
back_save_path = os.path.join(args.back_save_path, run_name)

trainer = DGSAN(writer=writer, device=device, dataset=args.dataset,
                init_embedd_dim=args.rnn_embedd_dim, init_hidden_size=args.rnn_hidden_size,
                init_learning_rate=args.learning_rate,
                dgsan_divergence_type=args.dgsan_divergence_type,
                learning_rate_gamma=args.learning_rate_gamma, batch_size=args.batch_size)

# DGSAN ######################################################################################
dgsan_step = DGSANStep(auto=True, loss_eps=args.loss_eps, max_epoch=args.step_max_epoch, min_epoch=args.step_min_epoch)

float_token_size = 1.
trainer.token_size = round(float_token_size)

for epoch in tqdm(range(args.epoch)):

    # DGSAN Step #############################################################################
    if dgsan_step.check(epoch):
        # _last_token_size = dgsan.token_size
        float_token_size = min(float_token_size + 0.2, trainer.length)
        trainer.token_size = round(float_token_size)
        # if _last_token_size != dgsan.token_size:
        #     dgsan.rst_lr()

        big_lr_detector = BigLRDetector()
        trainer.dgsan_step()

    trainer.write_parameters(epoch)
    trainer.model.train()
    loss_log = NamedStreamAverage()
    for batch_data in trainer.train_batchmanager:

        inp = batch_data.text.to(device)
        p_new_log_prob = trainer.model(sentences=inp)
        nll_loss = -1. * p_new_log_prob.mean()

        loss_log.add("Loss/NLLTrain", nll_loss.item() * inp.size(0), inp.size(0))

        if trainer.token_size >= inp.size(1):  # unconditional
            # P term #########################################################################
            with torch.no_grad():
                p_old_log_prob = trainer.old_model(sentences=inp)

            # Q term #########################################################################
            final_seq_len = inp.size(1)

            with torch.no_grad():
                inp = trainer.old_model.sample(number=inp.size(0), seq_len=final_seq_len,
                                               first_token=trainer.first_token,
                                               device=device)

                q_old_log_prob = trainer.old_model(sentences=inp)
            q_new_log_prob = trainer.model(sentences=inp)

        else:
            condition_len = random.randint(1, inp.size(1) - trainer.token_size)
            condition = inp[:, :condition_len]

            # P term #########################################################################
            inp = inp[:, condition_len:condition_len + trainer.token_size]

            with torch.no_grad():
                p_old_log_prob = trainer.old_model(condition=condition, sentences=inp)
            p_new_log_prob = trainer.model(condition=condition, sentences=inp)

            # Q term #########################################################################
            with torch.no_grad():
                inp = trainer.old_model.conditional_sample(condition=condition, extend_len=trainer.token_size,
                                                           temperature=0.5)  # the temperature in the code is reverse of standard definition.
                q_old_log_prob = trainer.old_model(condition=condition, sentences=inp)
            q_new_log_prob = trainer.model(condition=condition, sentences=inp)

        # Loss ###############################################################################
        loss = trainer.dgsan_loss(p_new_log_prob=p_new_log_prob, p_old_log_prob=p_old_log_prob,
                                  q_new_log_prob=q_new_log_prob, q_old_log_prob=q_old_log_prob)

        loss_log.add("Loss/DGSAN", loss.item() * inp.size(0), inp.size(0))

        inp = condition = p_new_log_prob = p_old_log_prob = q_new_log_prob = q_old_log_prob = None

        # Optimization #######################################################################
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

    dgsan_step.add_loss(loss_log["Loss/DGSAN"])

    # Dynamic LR #############################################################################
    big_lr_detector.add_loss(loss_log["Loss/DGSAN"])
    if big_lr_detector.check(epoch):
        print("LR is decreased")
        trainer.dec_lr()

    # Evaluate ###############################################################################
    if epoch % 10 == 0:
        trainer.model.eval()
        trainer.evaluation_phase(epoch, loss_log, back_save_path, samples_path)

trainer.save_last(save_path)

# free unnecessary resources #################################################################
trainer.old_model = trainer.model = trainer.optimizer = trainer.scheduler = None
trainer.train_batchmanager = trainer.val_batchmanager = trainer.test_batchmanager = None
torch.cuda.empty_cache()

# finish parallel threads of evaluation ######################################################
trainer.finish_evaluation()

# select best epoch based on validation ######################################################
if trainer.evaluate_valid:
    trainer.select_best_save(back_save_path, save_path)
