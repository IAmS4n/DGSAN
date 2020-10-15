import queue
import time

import torch.multiprocessing as mp

from metrics.bert_distances import FBD_EMBD
from metrics.ngram_metrics import Jaccard, Bleu, SelfBleu, NgramProp


def _bert_eval(references, inp_q, out_q, set_name, silent=True):
    fbd_embd_metr = FBD_EMBD(references=references, model_name="bert-base-uncased",
                             bert_model_dir="./bert-data/")
    print("BERT is loaded.")

    while True:
        epoch, samples = inp_q.get(block=True)
        res = {"BERT_%s/%s" % (set_name, name): val for name, val in
               fbd_embd_metr.get_score(sentences=samples).items()}
        out_q.put((epoch, res))


def _ngram_eval(references, TEXT, inp_q, out_q, set_name, silent=True):
    jaccard_metr = Jaccard(references=references, parser=TEXT, parse=False)
    bleu_metr = Bleu(references=references, parser=TEXT, parse=False)

    while True:
        epoch, samples = inp_q.get(block=True)

        jaccard_value = jaccard_metr.eval(samples, parse=False)
        blue_value = bleu_metr.eval(samples, parse=False)
        self_bleu_value = SelfBleu(samples=samples, parser=TEXT, parse=False).eval()
        ngram_prop = NgramProp(samples=samples, parser=TEXT, parse=False).eval()

        res = {}
        for n in jaccard_value:
            res["ngrams_%s/Jaccard_%s" % (set_name, n)] = jaccard_value[n]
        for n in blue_value:
            res["ngrams_%s/BLEU_%s" % (set_name, n)] = blue_value[n]
        for n in self_bleu_value:
            res["ngrams_%s/SelfBLEU_%s" % (set_name, n)] = self_bleu_value[n]
        for prop_name in ngram_prop:
            res["ngrams_%s/Prop_%s" % (set_name, prop_name)] = ngram_prop[prop_name]

        out_q.put((epoch, res))


class MetricsEval:
    def __init__(self, test_ds, TEXT, set_name):
        self.bests = {}

        self.to_bert_process = mp.Queue()
        self.to_ngram_process = mp.Queue()

        self.from_process = mp.Queue()

        references = list(TEXT.detokenize(test_ds.text))  # convert tokenized sentences to normal sentences

        self.bert_process = mp.Process(target=_bert_eval,
                                       args=(references, self.to_bert_process, self.from_process, set_name))
        self.bert_process.daemon = True
        self.bert_process.start()

        self.ngram_process = mp.Process(target=_ngram_eval,
                                        args=(references, TEXT, self.to_ngram_process, self.from_process, set_name))
        self.ngram_process.daemon = True
        self.ngram_process.start()

    def add(self, epoch, samples, bert=True):
        self.to_ngram_process.put((epoch, samples))
        if bert:
            self.to_bert_process.put((epoch, samples))

    def get(self, block=False):
        try:
            eval_epoch, eval_res = self.from_process.get(block=block, timeout=60)
            for eval_res_name in eval_res:
                must_update = False

                if eval_res_name not in self.bests:
                    must_update = True
                elif ("bert" in eval_res_name.lower()) or ("SelfBLEU" in eval_res_name.lower()):
                    if eval_res[eval_res_name] < self.bests[eval_res_name][1]:
                        must_update = True
                else:
                    if eval_res[eval_res_name] > self.bests[eval_res_name][1]:
                        must_update = True

                if must_update:
                    self.bests[eval_res_name] = (eval_epoch, eval_res[eval_res_name])

            return eval_epoch, eval_res
        except queue.Empty:
            return None

    def update_writer(self, writer, block=False):
        while True:
            process_res = self.get(block=block)
            if process_res is None:
                return

            eval_epoch, eval_res = process_res
            for eval_res_name in eval_res:
                writer.add_scalar(eval_res_name, eval_res[eval_res_name], eval_epoch)

    def finish(self, writer):
        while (not self.to_ngram_process.empty()) or (not self.to_bert_process.empty()):
            self.update_writer(writer=writer, block=True)
            time.sleep(5 * 60)

    def print_status(self):
        print("to_ngram_process", self.to_ngram_process.qsize())
        print("to_bert_process", self.to_bert_process.qsize())
        print("from_process", self.from_process.qsize())
