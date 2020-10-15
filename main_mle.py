import argparse
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainer import Trainer
from utils import NamedStreamAverage

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(prog='MLE')
parser.add_argument('--batch_size', type=int, required=False, default=128)
parser.add_argument('--dataset', type=str, required=False, default="")
parser.add_argument('--details', type=str, required=False, default="")

parser.add_argument('--epoch', type=int, required=False, default=1000)

parser.add_argument('--log_path', type=str, required=False, default="./log")
parser.add_argument('--samples_path', type=str, required=False, default="./samples")
parser.add_argument('--save_path', type=str, required=False, default="./save")
parser.add_argument('--back_save_path', type=str, required=False, default="./back_save")

parser.add_argument('--rnn_embedd_dim', type=int, required=False, default=128)
parser.add_argument('--rnn_hidden_size', type=int, required=False, default=64)
parser.add_argument('--learning_rate', type=float, required=False, default=1e-3)
parser.add_argument('--learning_rate_gamma', type=float, required=False, default=0.5)

args = parser.parse_args()

os.makedirs(args.log_path, exist_ok=True)
os.makedirs(args.samples_path, exist_ok=True)
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(args.back_save_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device', device)

# log #########################################################################################
data_abs = args.dataset
net_abs = "RNN%d-%d" % (args.rnn_embedd_dim, args.rnn_hidden_size)
train_abs = "MLE"

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

trainer = Trainer(writer=writer, device=device, dataset=args.dataset,
                  init_embedd_dim=args.rnn_embedd_dim, init_hidden_size=args.rnn_hidden_size,
                  init_learning_rate=args.learning_rate, learning_rate_gamma=args.learning_rate_gamma,
                  batch_size=args.batch_size, )

for epoch in tqdm(range(args.epoch)):
    trainer.write_parameters(epoch)
    trainer.model.train()
    loss_log = NamedStreamAverage()

    for batch_data in trainer.train_batchmanager:
        inp = batch_data.text.to(device)
        p_new_log_prob = trainer.model(sentences=inp)
        nll_loss = -1. * p_new_log_prob.mean()

        loss_log.add("Loss/NLLTrain", nll_loss.item() * inp.size(0), inp.size(0))

        trainer.optimizer.zero_grad()
        nll_loss.backward()
        trainer.optimizer.step()

    # Dynamic LR ##############################################################################
    if epoch > 0 and epoch % 100 == 0:
        print("LR is decreased")
        trainer.dec_lr()

    # Evaluate ################################################################################
    if epoch % 10 == 0:
        trainer.model.eval()
        trainer.evaluation_phase(epoch, loss_log, back_save_path, samples_path)

trainer.save_last(save_path)

# free unnecessary resources ##################################################################
trainer.model = trainer.optimizer = trainer.scheduler = None
trainer.train_batchmanager = trainer.val_batchmanager = trainer.test_batchmanager = None
torch.cuda.empty_cache()

# finish parallel threads of evaluation #######################################################
trainer.finish_evaluation()

# select best epoch based on validation #######################################################
if trainer.evaluate_valid:
    trainer.select_best_save(back_save_path, save_path)
