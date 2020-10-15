import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self._word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self._lstm_model = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self._hidden2logit = nn.Linear(hidden_dim, tagset_size)
        self._selector = nn.NLLLoss(reduction="none")

    def _lstm(self, sentences, temperature=1.):
        bs, sl = sentences.size()
        embeds = self._word_embeddings(sentences)
        # assert embeds.size() == (bs, sl, self.embedding_dim)

        lstm_out, _ = self._lstm_model(embeds.transpose(0, 1))
        lstm_out = lstm_out.transpose(0, 1)
        # assert lstm_out.size() == (bs, sl, self.hidden_dim)

        lstm_out = lstm_out.reshape((bs * sl, self.hidden_dim))
        logit = self._hidden2logit(lstm_out)
        logit = logit.view((bs, sl, self.vocab_size))

        log_prob = F.log_softmax(logit * temperature, dim=-1)

        return log_prob

    def _next_lstm(self, tokens, state):
        tokens = tokens.view(-1, 1)  # sl = 1
        embeds = self._word_embeddings(tokens)
        # assert embeds.size() == (bs, 1, self.embedding_dim)

        lstm_out, state = self._lstm_model(embeds.transpose(0, 1), hx=state)
        # lstm_out = lstm_out.transpose(0, 1)
        # assert lstm_out.size() == (bs, 1, self.hidden_dim)

        lstm_out = lstm_out.reshape((-1, self.hidden_dim))
        logit = self._hidden2logit(lstm_out)

        return logit, state

    def _sampling(self, token, state, gen_len, temperature=1.):
        res = [token, ]
        for _ in range(gen_len):
            logit, state = self._next_lstm(tokens=token, state=state)
            prediction_vector = F.softmax(logit * temperature, dim=1)
            del logit
            token = torch.multinomial(prediction_vector, 1)[:, 0]
            del prediction_vector
            res.append(token)

        return torch.stack(res, 1).detach()

    def sample(self, number, seq_len, first_token, device, temperature=1.):
        token = torch.tensor([first_token] * number, device=device)
        state_h = torch.zeros(1, number, self.hidden_dim, device=device)
        state_c = torch.zeros(1, number, self.hidden_dim, device=device)
        state = (state_h, state_c)

        return self._sampling(token=token, state=state, gen_len=seq_len, temperature=temperature)

    def conditional_sample(self, condition, extend_len, temperature=1.):
        assert extend_len > 0
        bs, sl = condition.size()
        embeds = self._word_embeddings(condition)
        assert embeds.size() == (bs, sl, self.embedding_dim)

        lstm_out, state = self._lstm_model(embeds.transpose(0, 1))
        del embeds
        lstm_out = lstm_out[-1]
        assert lstm_out.size() == (bs, self.hidden_dim)

        logit = self._hidden2logit(lstm_out)
        del lstm_out

        prediction_vector = F.softmax(logit, dim=-1)
        del logit
        token = torch.multinomial(prediction_vector, 1)[:, 0]

        return self._sampling(token=token, state=state, gen_len=extend_len - 1, temperature=temperature)

    def forward(self, sentences, condition=None, temperature=1.):  # get prob
        if condition is None:
            # sentences: batch * len
            # output: batch

            log_prob_per_vocab = self._lstm(sentences[:, :-1], temperature=temperature)
            target = sentences[:, 1:].detach()
        else:
            assert condition.size(0) == sentences.size(0)
            # condition, sentences: batch * len
            # output: batch

            all_sentences = torch.cat([condition, sentences[:, :-1]], 1)
            log_prob_per_vocab = self._lstm(all_sentences, temperature=temperature)[:, -1 * sentences.size(1):, :]
            target = sentences.detach()

        conditional_log_prob = -1. * self._selector(log_prob_per_vocab.transpose(-1, -2), target)
        # ln p(x1, x2, ...) = ln p(x1) + ln p(x2|x1) + ln p(x3| x1, x2) + ...
        log_prob = conditional_log_prob.sum(1)
        return log_prob
