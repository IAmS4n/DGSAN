from abc import abstractmethod
from collections import Counter
from functools import reduce

import numpy as np
from nltk.translate.bleu_score import ngrams
from torchtext.data import ReversibleField


def get_ngrams(sentences, n):
    return [list(ngrams(x, n)) if len(x) >= n else [] for x in sentences]


class Metric:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def eval(self, *args):
        pass


class Jaccard(Metric):
    def __init__(self, references, min_n=2, max_n=5, parser: ReversibleField = None, parse=True):
        super().__init__('jaccard')
        print('multiset distances init upto {}!'.format(max_n))
        if parse:
            references = parser.reverse(references)
        references = [parser.tokenize(r) for r in references]
        self.references = references
        self.max_n = max_n
        self.min_n = min_n
        self.parser = parser
        assert self.max_n >= self.min_n
        assert self.min_n >= 1
        self.ref_ngrams = self._get_ngrams(references)
        print('jaccard instance created!')

    def _get_ngrams(self, samples):
        samples_size = len(samples)
        all_counters = [Counter([x for y in get_ngrams(samples, n + 1) for x in y])
                        for n in range(self.max_n)]
        for n_counter in all_counters:
            for k in n_counter.keys():
                n_counter[k] /= samples_size
        return all_counters

    def eval(self, samples, parse=True):
        if parse:
            samples = self.parser.reverse(samples)
        samples = [self.parser.tokenize(r) for r in samples]
        print('multiset distances preprocess upto {}!'.format(self.max_n))
        sample_ngrams = self._get_ngrams(samples)
        ngrams_intersection = [sample_ngrams[i] & self.ref_ngrams[i]
                               for i in range(self.max_n)]  # intersection:  min(c[x], d[x])
        ngrams_union = [sample_ngrams[i] | self.ref_ngrams[i]
                        for i in range(self.max_n)]  # union:  max(c[x], d[x])
        temp_results = {}

        temp_results['jaccard'] = [float(sum(ngrams_intersection[n].values())) /
                                   sum(ngrams_union[n].values())
                                   for n in range(self.max_n)]
        result = {}
        key = 'jaccard'
        for n in range(self.min_n, self.max_n + 1):
            result[n] = np.power(
                reduce(lambda x, y: x * y, temp_results[key][:n]), 1. / n)
        return result


class Bleu(Metric):
    def __init__(self, references, parser: ReversibleField = None, parse=True):
        super().__init__('bleu')
        if parse:
            references = parser.reverse(references)
        ref_tokens = [parser.tokenize(r) for r in references]
        self.parser = parser
        from fast_bleu import BLEU as FBLEU
        w = {i: np.ones(i) / i for i in range(2, 6)}
        self.bleu = FBLEU(ref_tokens, w)
        print('bleu instance created!')

    def eval(self, samples, parse=True):
        if parse:
            samples = self.parser.reverse(samples)
        samples = [self.parser.tokenize(r) for r in samples]
        scores = self.bleu.get_score(samples)
        return {k: np.mean(scores[k]) for k in scores.keys()}


class SelfBleu(Metric):
    def __init__(self, samples, parser: ReversibleField = None, parse=True):
        super().__init__('self-bleu')
        if parse:
            samples = parser.reverse(samples)
        ref_tokens = [parser.tokenize(r) for r in samples]
        from fast_bleu import SelfBLEU as FSBLEU
        w = {i: np.ones(i) / i for i in range(2, 6)}
        self.bleu = FSBLEU(ref_tokens, w)
        print('self-bleu instance created!')

    def eval(self):
        scores = self.bleu.get_score()
        return {k: np.mean(scores[k]) for k in scores.keys()}


class NgramProp(Metric):
    def __init__(self, samples, parser: ReversibleField = None, parse=True):
        super().__init__('NgramProp')
        if parse:
            samples = parser.reverse(samples)
        ref_tokens = [parser.tokenize(r) for r in samples]
        scores = []
        lens = []
        for sample in ref_tokens:
            if len(sample) == 0:
                continue
            smaple_score = float(len(Counter(sample))) / float(len(sample))
            scores.append(smaple_score)
            lens.append(len(sample))
        self.score = float(np.mean(scores))
        self.len = float(np.mean(lens))

    def eval(self):
        return {"Unique_1": self.score, "Len": self.len}
