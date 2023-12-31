# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import OrderedDict
from logging import getLogger

import numpy as np
import torch

from ..utils import concat_batches, to_cuda

logger = getLogger()


class Evaluator(object):
    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.dico = data["dico"]
        self.params = params

    def get_iterator(self, data_set, lang1, lang2=None, stream=False):
        """
        Create a new iterator for a dataset.
        """
        assert data_set in ["valid", "test"]
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None

        # hacks to reduce evaluation time when using many languages
        n_sentences = -1
        subsample = 1

        if lang2 is None:
            assert stream is True
            iterator = self.data["mono_stream"][lang1][data_set].get_iterator(
                shuffle=False, subsample=subsample
            )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data["para"][(_lang1, _lang2)][data_set].get_iterator(
                shuffle=False, group_by_size=True, n_sentences=n_sentences
            )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 else batch[::-1]

    def mask_out(self, x, lengths, rng):
        """
        Decide of random words to mask out.
        We specify the random generator to ensure that the test is the same at each epoch.
        """
        params = self.params
        slen, bs = x.size()

        # words to predict - be sure there is at least one word per sentence
        to_predict = rng.rand(slen, bs) <= params.word_pred
        to_predict[0] = 0
        for i in range(bs):
            to_predict[lengths[i] - 1 :, i] = 0
            if not np.any(to_predict[: lengths[i] - 1, i]):
                v = rng.randint(1, lengths[i] - 1)
                to_predict[v, i] = 1
        pred_mask = torch.from_numpy(to_predict.astype(np.uint8))

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_mask = _x_real.clone().fill_(params.mask_index)
        x = x.masked_scatter(pred_mask, _x_mask)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({"epoch": trainer.epoch})

        with torch.no_grad():

            for data_set in ["valid", "test"]:

                # prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.mlm_steps:
                    self.evaluate_mlm(scores, data_set, lang1, lang2)

                # report average metrics per language
                _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
                if len(_mlm_mono) > 0:
                    scores["%s_mlm_ppl" % data_set] = np.mean(
                        [
                            scores["%s_%s_mlm_ppl" % (data_set, lang)]
                            for lang in _mlm_mono
                        ]
                    )
                    scores["%s_mlm_acc" % data_set] = np.mean(
                        [
                            scores["%s_%s_mlm_acc" % (data_set, lang)]
                            for lang in _mlm_mono
                        ]
                    )

        return scores

    def evaluate_mlm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ["valid", "test"]
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model
        model.eval()
        model = model.module if params.multi_gpu else model

        rng = np.random.RandomState(0)

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None
        l1l2 = lang1 if lang2 is None else f"{lang1}_{lang2}"

        n_words = 0
        xe_loss = 0
        n_valid = 0

        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None)):

            # batch
            if lang2 is None:
                x, lengths = batch
                positions = None
                langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            else:
                (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(
                    sent1,
                    len1,
                    lang1_id,
                    sent2,
                    len2,
                    lang2_id,
                    params.pad_index,
                    params.eos_index,
                    reset_positions=True,
                )

            # words to predict
            x, y, pred_mask = self.mask_out(x, lengths, rng)

            # cuda
            x, y, pred_mask, lengths, positions, langs = to_cuda(
                x, y, pred_mask, lengths, positions, langs
            )

            # forward / loss
            tensor = model(
                "fwd",
                x=x,
                lengths=lengths,
                positions=positions,
                langs=langs,
            )
            word_scores, loss = model(
                "predict", tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True
            )

            # update stats
            n_words += len(y)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        ppl_name = "%s_%s_mlm_ppl" % (data_set, l1l2)
        acc_name = "%s_%s_mlm_acc" % (data_set, l1l2)
        scores[ppl_name] = np.exp(xe_loss / n_words) if n_words > 0 else 1e9
        scores[acc_name] = 100.0 * n_valid / n_words if n_words > 0 else 0.0


class SingleEvaluator(Evaluator):
    def __init__(self, trainer, data, params):
        """
        Build language model evaluator.
        """
        super().__init__(trainer, data, params)
        self.model = trainer.model
