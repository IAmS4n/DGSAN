import torch
import torch.nn as nn


def get_loss_func(name):
    loss_p_term = loss_q_term = None
    if name == "JS":
        softplus1 = nn.Softplus()
        softplus2 = nn.Softplus()
        loss_p_term = lambda model_logit, old_model_logit: -1. * softplus1(old_model_logit - model_logit)
        loss_q_term = lambda model_logit, old_model_logit: -1. * softplus2(model_logit - old_model_logit)
    elif name == "KL":
        loss_p_term = lambda model_logit, old_model_logit: model_logit - old_model_logit
        loss_q_term = lambda model_logit, old_model_logit: -1. * torch.exp(model_logit - old_model_logit)
    elif name == "RKL":
        loss_p_term = lambda model_logit, old_model_logit: -1. * torch.exp(old_model_logit - model_logit)
        loss_q_term = lambda model_logit, old_model_logit: old_model_logit - model_logit
    elif name == "PX2":
        loss_p_term = lambda model_logit, old_model_logit: 2. * torch.exp(model_logit - old_model_logit)
        loss_q_term = lambda model_logit, old_model_logit: torch.exp(2. * (model_logit - old_model_logit))
    elif name == "SH":
        loss_p_term = lambda model_logit, old_model_logit: -1. * torch.exp(0.5 * (old_model_logit - model_logit))
        loss_q_term = lambda model_logit, old_model_logit: -1. * torch.exp(0.5 * (model_logit - old_model_logit))
    elif name == "NX2":
        loss_p_term = lambda model_logit, old_model_logit: -1. * torch.exp(2. * (old_model_logit - model_logit))
        loss_q_term = lambda model_logit, old_model_logit: 2. * torch.exp(old_model_logit - model_logit)
    elif name == "Jeff":
        loss_p_term = lambda model_logit, old_model_logit: (model_logit - old_model_logit) - torch.exp(
            old_model_logit - model_logit)
        loss_q_term = lambda model_logit, old_model_logit: -(model_logit - old_model_logit) - torch.exp(
            model_logit - old_model_logit)
    assert (loss_p_term is not None) and (loss_q_term is not None)
    return loss_p_term, loss_q_term


class DGSANLoss(nn.Module):
    def __init__(self, type):
        super(DGSANLoss, self).__init__()
        self.p_term_func, self.q_term_func = get_loss_func(type)

    def forward(self, p_new_log_prob, p_old_log_prob, q_new_log_prob, q_old_log_prob):
        # log prob : batch
        assert p_new_log_prob.size() == p_old_log_prob.size()
        assert q_new_log_prob.size() == q_old_log_prob.size()
        assert len(p_new_log_prob.size()) == len(q_new_log_prob.size()) == 1

        loss1 = -self.p_term_func(model_logit=p_new_log_prob, old_model_logit=p_old_log_prob.detach()).mean()
        loss2 = -self.q_term_func(model_logit=q_new_log_prob, old_model_logit=q_old_log_prob.detach()).mean()

        return loss1 + loss2

# class DGSANLossEff(nn.Module):
#     def __init__(self, type):
#         super(DGSANLossEff, self).__init__()
#         self.p_term_func, self.q_term_func = get_loss_func(type)
#
#     def forward(self, p_new_log_prob=None, p_old_log_prob=None, q_new_log_prob=None, q_old_log_prob=None):
#         # log prob : batch
#
#         loss1 = loss2 = None
#         if (p_new_log_prob is not None) and (p_old_log_prob is not None):
#             assert p_new_log_prob.size() == p_old_log_prob.size()
#             assert len(p_new_log_prob.size()) == 1
#             loss1 = -self.p_term_func(model_logit=p_new_log_prob, old_model_logit=p_old_log_prob).mean()
#         if (q_new_log_prob is not None) and (q_old_log_prob is not None):
#             assert q_new_log_prob.size() == q_old_log_prob.size()
#             assert len(q_new_log_prob.size()) == 1
#             loss2 = -self.q_term_func(model_logit=q_new_log_prob, old_model_logit=q_old_log_prob).mean()
#
#         if (loss1 is not None) and (loss2 is not None):
#             return loss1 + loss2
#         elif loss1 is not None:
#             return loss1
#         elif loss2 is not None:
#             return loss2
#         else:
#             raise

# class DGSANSeqLoss(nn.Module):
#     def __init__(self, type):
#         super(DGSANSeqLoss, self).__init__()
#         self.dgsan_loss = DGSANLoss(type=type)
#         self.selector = nn.NLLLoss(reduction="none")
#
#     def forward(self, p_new_log_prob, p_old_log_prob, p_taqrget,
#                 q_new_log_prob, q_old_log_prob, q_target):
#         # log prob : batch * len * vocab
#         # target : batch * len
#         assert p_new_log_prob.size() == p_old_log_prob.size()
#         assert q_new_log_prob.size() == q_old_log_prob.size()
#         assert len(p_new_log_prob.size()) == len(q_new_log_prob.size()) == 3
#
#         p_new_log_prob = -1. * self.selector(p_new_log_prob.transpose(-1, -2), p_taqrget).sum(1)
#         p_old_log_prob = -1. * self.selector(p_old_log_prob.transpose(-1, -2).detach(), p_taqrget).sum(1)
#
#         q_new_log_prob = -1. * self.selector(q_new_log_prob.transpose(-1, -2), q_target).sum(1)
#         q_old_log_prob = -1. * self.selector(q_old_log_prob.transpose(-1, -2).detach(), q_target).sum(1)
#
#         loss = self.dgsan_loss(p_new_log_prob, p_old_log_prob, q_new_log_prob, q_old_log_prob, )
#         return loss
