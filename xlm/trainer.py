# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
from collections import OrderedDict
from logging import getLogger
from argparse import Namespace
from typing import Dict, Tuple, Union

import apex
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .model.transformer import TransformerFFN, TransformerModel
from .optim import get_optimizer
from .utils import concat_batches, find_modules, to_cuda
from xlm.data.dataset import ParallelDataset, StreamDataset

logger = getLogger()
type_dataset = Dict[str, Dict[Union[str, Tuple[str, str]], Dict[str, Union[StreamDataset, ParallelDataset]]]]


class Trainer(object):
    def __init__(self, data: type_dataset, params: Namespace):
        """
        Initialize trainer.
        """
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # list memory components
        self.ffn_list = []
        find_modules(self.model, "self.model", TransformerFFN, self.ffn_list)
        logger.info("Found %i FFN." % len(self.ffn_list))

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        if params.multi_gpu and params.amp == -1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[params.local_rank],
                output_device=params.local_rank,
                broadcast_buffers=True,
            )

        # set optimizers
        self.set_optimizers()

        # float16 / distributed (AMP)
        if params.amp >= 0:
            self.init_amp()
            if params.multi_gpu:
                logger.info("Using apex.parallel.DistributedDataParallel ...")
                self.model = apex.parallel.DistributedDataParallel(
                    self.model, delay_allreduce=True
                )

        # stopping criterion used for early stopping
        if params.stopping_criterion != "":
            split = params.stopping_criterion.split(",")
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == "_":
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # probability of masking out / randomize / not modify words to predict
        params.pred_probs = torch.FloatTensor(
            [params.word_mask, params.word_keep, params.word_rand]
        )

        # probabilty to predict a word
        sample_alpha = 0
        counts = np.array(list(self.data["dico"].counts.values()))
        params.mask_scores = np.maximum(counts, 1) ** -sample_alpha
        params.mask_scores[params.pad_index] = 0  # do not predict <PAD> index
        params.mask_scores[counts == 0] = 0  # do not predict special tokens

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m[1:], False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {
            metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics
        }

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        self.stats = OrderedDict(
            [("processed_s", 0), ("processed_w", 0)]
            + [("MLM-%s" % l, []) for l in params.langs]
            + [("MLM-%s-%s" % (l1, l2), []) for l1, l2 in data["para"].keys()]
            + [("MLM-%s-%s" % (l2, l1), []) for l1, l2 in data["para"].keys()]
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

    def set_parameters(self) -> None:
        """
        Set parameters.
        """
        named_params = []
        named_params.extend(
            [(k, p) for k, p in self.model.named_parameters() if p.requires_grad]
        )

        # model (excluding memory values)
        # named_params has only model value.s
        self.parameter = [p for k, p in named_params]

    def set_optimizers(self) -> None:
        """
        Set optimizers.
        """
        params = self.params

        # model optimizer (excluding memory values)
        self.optimizer = get_optimizer(self.parameter, params.optimizer)

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert (
            params.amp == 0
            and params.fp16 is False
            or params.amp in [1, 2, 3]
            and params.fp16 is True
        )

        self.model, self.optimizer = apex.amp.initialize(
            self.model,
            self.optimizer,
            opt_level=("O%i" % params.amp),
        )

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        params = self.params

        # regular optimization
        if params.amp == -1:
            self.optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                clip_grad_norm_(self.parameter, params.clip_grad_norm)
            self.optimizer.step()

        # AMP optimization
        else:
            if self.n_iter % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    clip_grad_norm_(
                        apex.amp.master_params(self.optimizer),
                        params.clip_grad_norm,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(
                    loss, self.optimizer, delay_unscale=True
                ) as scaled_loss:
                    scaled_loss.backward()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 5 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = " || ".join(
            [
                "{}: {:7.4f}".format(k, np.mean(v))
                for k, v in self.stats.items()
                if type(v) is list and len(v) > 0
            ]
        )
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = " - "
        s_lr = (
            s_lr
            + (" - %s LR: " % k)
            + " / ".join(
                "{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups
            )
        )

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats["processed_s"] * 1.0 / diff,
            self.stats["processed_w"] * 1.0 / diff,
        )
        self.stats["processed_s"] = 0
        self.stats["processed_w"] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def get_iterator(self, iter_name, lang1, lang2, stream):
        """
        Create a new iterator for a dataset.
        """
        logger.info(
            "Creating new training data iterator (%s) ..."
            % ",".join([str(x) for x in [iter_name, lang1, lang2] if x is not None])
        )
        if lang2 is None:
            assert stream is True
            iterator = self.data["mono_stream"][lang1]["train"].get_iterator(
                shuffle=True
            )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data["para"][(_lang1, _lang2)]["train"].get_iterator(
                shuffle=True,
                group_by_size=self.params.group_by_size,
                n_sentences=-1,
            )

        self.iterators[(iter_name, lang1, lang2)] = iterator
        return iterator

    def get_batch(self, iter_name, lang1, lang2=None, stream=False):
        """
        Return a batch of sentences from a dataset.
        """
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None
        iterator = self.iterators.get((iter_name, lang1, lang2), None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream)
            x = next(iterator)
        return x if lang2 is None or lang1 < lang2 else x[::-1]

    def mask_out(self, x):
        """
        Decide of random words to mask out, and what target they get assigned.
        """
        params = self.params
        slen, bs = x.size()

        # define target words to predict
        pred_mask = np.random.rand(slen, bs) <= params.word_pred
        pred_mask = torch.from_numpy(pred_mask.astype(np.uint8))

        # do not predict padding
        pred_mask[x == params.pad_index] = 0
        pred_mask[0] = 0  # TODO: remove

        # mask a number of words == 0 [8] (faster with fp16)
        if params.fp16:
            pred_mask = pred_mask.view(-1)
            n1 = pred_mask.sum().item()
            n2 = max(n1 % 8, 8 * (n1 // 8))
            if n2 != n1:
                pred_mask[torch.nonzero(pred_mask).view(-1)[: n1 - n2]] = 0
            pred_mask = pred_mask.view(slen, bs)
            assert pred_mask.sum().item() % 8 == 0

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_rand = _x_real.clone().random_(params.n_words)
        _x_mask = _x_real.clone().fill_(params.mask_index)
        probs = torch.multinomial(params.pred_probs, len(_x_real), replacement=True)
        _x = (
            _x_mask * (probs == 0).long()
            + _x_real * (probs == 1).long()
            + _x_rand * (probs == 2).long()
        )
        x = x.masked_scatter(pred_mask, _x)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def generate_batch(self, lang1, lang2, name):
        """
        Prepare a batch (for causal or non-causal mode).
        """
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None

        if lang2 is None:
            x, lengths = self.get_batch(name, lang1, stream=True)
            positions = None
            langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
        else:
            (x1, len1), (x2, len2) = self.get_batch(name, lang1, lang2)
            x, lengths, positions, langs = concat_batches(
                x1,
                len1,
                lang1_id,
                x2,
                len2,
                lang2_id,
                params.pad_index,
                params.eos_index,
                reset_positions=True,
            )

        return (
            x,
            lengths,
            positions,
            langs,
            (None, None) if lang2 is None else (len1, len2),
        )

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "best_stopping_criterion": self.best_stopping_criterion,
        }

        logger.warning("Saving model parameters ...")
        data["model"] = self.model.state_dict()

        if include_optimizers:
            logger.warning("Saving model optimizer ...")
            data["model_optimizer"] = self.optimizer.state_dict()

        data["dico_id2word"] = self.data["dico"].id2word
        data["dico_word2id"] = self.data["dico"].word2id
        data["dico_counts"] = self.data["dico"].counts
        data["params"] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def reload_checkpoint(self) -> None:
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, "checkpoint.pth")
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == "":
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu")

        # reload model parameters
        ##
        # model = data["model"]
        # model = {"module." + k: v for k, v in model.items()}
        # self.model.load_state_dict(model)
        ##
        self.model.load_state_dict(data["model"])
        # reload optimizers
        logger.warning("Not reloading checkpoint optimizer model.")
        for group_id, param_group in enumerate(self.optimizer.param_groups):
            if "num_updates" not in param_group:
                logger.warning("No 'num_updates' for optimizer model.")
                continue
            logger.warning("Reloading 'num_updates' and 'lr' for optimizer model.")
            param_group["num_updates"] = data["model_optimizer"]["param_groups"][
                group_id
            ]["num_updates"]
            param_group["lr"] = self.optimizer.get_lr_for_step(
                param_group["num_updates"]
            )

        # reload main metrics
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.best_metrics = data["best_metrics"]
        self.best_stopping_criterion = data["best_stopping_criterion"]
        logger.warning(
            f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ..."
        )

    def save_periodic(self):
        """
        Save the models periodically.
        """
        save_epoch = [int(epoch.strip()) for epoch in self.params.save_epoch.split(",")]
        if not self.params.is_master:
            return
        if (
            self.params.save_periodic > 0
            and self.epoch % self.params.save_periodic == 0
        ):
            self.save_checkpoint("periodic-%i" % self.epoch, include_optimizers=False)
        # elif self.epoch in [0, 1, 2, 3, 4, 9, 19, 29, 39, 49, 99]:
        #     self.save_checkpoint("periodic-%i" % self.epoch, include_optimizers=False)
        elif self.epoch in save_epoch:
            self.save_checkpoint("periodic-%i" % self.epoch, include_optimizers=False)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[metric]))
                self.save_checkpoint("best-%s" % metric, include_optimizers=False)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None:
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info(
                    "New best validation score: %f" % self.best_stopping_criterion
                )
                self.decrease_counts = 0
            else:
                logger.info(
                    "Not a better validation score (%i / %i)."
                    % (self.decrease_counts, self.decrease_counts_max)
                )
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info(
                    "Stopping criterion has been below its best value for more "
                    "than %i epochs. Ending the experiment..."
                    % self.decrease_counts_max
                )
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()
        self.save_checkpoint("checkpoint", include_optimizers=True)
        self.epoch += 1

    def round_batch(self, x, lengths, positions, langs):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding,
        so that each dimension is a multiple of 8.
        """
        params = self.params
        if not params.fp16 or len(lengths) < 8:
            return x, lengths, positions, langs, None

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[:slen, idx]
            positions = None if positions is None else positions[:slen, idx]
            langs = None if langs is None else langs[:slen, idx]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(0)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            x = torch.cat([x, torch.LongTensor(pad, bs2).fill_(params.pad_index)], 0)
            if positions is not None:
                positions = torch.cat(
                    [positions, torch.arange(pad)[:, None] + positions[-1][None] + 1], 0
                )
            if langs is not None:
                langs = torch.cat([langs, langs[-1][None].expand(pad, bs2)], 0)
            assert x.size() == (ml2, bs2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths, positions, langs, idx

    def mlm_step(self, lang1, lang2):
        """
        Masked word prediction step.
        MLM objective is lang2 is None, TLM objective otherwise.
        """
        params = self.params
        model = self.model
        model.train()

        # generate batch / select words to predict
        x, lengths, positions, langs, _ = self.generate_batch(lang1, lang2, "pred")
        x, lengths, positions, langs, _ = self.round_batch(x, lengths, positions, langs)
        x, y, pred_mask = self.mask_out(x)

        # cuda
        x, y, pred_mask, lengths, positions, langs = to_cuda(
            x, y, pred_mask, lengths, positions, langs
        )

        # forward / loss
        tensor = model("fwd", x=x, lengths=lengths, positions=positions, langs=langs)
        _, loss = model(
            "predict", tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False
        )
        self.stats[
            ("MLM-%s" % lang1) if lang2 is None else ("MLM-%s-%s" % (lang1, lang2))
        ].append(loss.item())

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats["processed_s"] += lengths.size(0)
        self.stats["processed_w"] += pred_mask.sum().item()


class SingleTrainer(Trainer):
    def __init__(self, model: TransformerModel, data: type_dataset, params: Namespace):

        # model / data / params
        self.model = model
        self.data = data
        self.params = params

        super().__init__(data, params)
