# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import torch
import torch.nn as nn

from ..data.dictionary import (
    BOS_WORD,
    EOS_WORD,
    MASK_WORD,
    PAD_WORD,
    UNK_WORD,
    Dictionary,
)
from ..utils import AttrDict
from .transformer import TransformerModel

logger = getLogger()


class SentenceEmbedder(object):
    @staticmethod
    def reload(path, params):
        """
        Create a sentence embedder from a pretrained model.
        """
        # reload model
        reloaded = torch.load(path)
        state_dict = reloaded["model"]

        # reload dictionary and model parameters
        dico = Dictionary(
            reloaded["dico_id2word"], reloaded["dico_word2id"], reloaded["dico_counts"]
        )
        pretrain_params = AttrDict(reloaded["params"])
        pretrain_params.n_words = len(dico)
        pretrain_params.bos_index = dico.index(BOS_WORD)
        pretrain_params.eos_index = dico.index(EOS_WORD)
        pretrain_params.pad_index = dico.index(PAD_WORD)
        pretrain_params.unk_index = dico.index(UNK_WORD)
        pretrain_params.mask_index = dico.index(MASK_WORD)

        # build_modelの代わり
        # build model and reload weights
        model = TransformerModel(pretrain_params, dico)
        # model.load_state_dict(state_dict)
        # model.load_state_dict({k[len("module.") :]: v for k, v in state_dict.items()})
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in state_dict.items()}
        )
        model.cuda()
        model.eval()

        return SentenceEmbedder(model, dico, pretrain_params, params)

    def __init__(self, model, dico, pretrain_params, params):
        """
        Wrapper on top of the different sentence embedders.
        Returns sequence-wise or single-vector sentence representations.
        """
        self.pretrain_params = {k: v for k, v in pretrain_params.__dict__.items()}
        self.model = model
        self.dico = dico
        self.n_layers = model.n_layers
        self.out_dim = model.dim
        self.n_words = model.n_words
        # self.model.to("cuda")
        if params.multi_gpu:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[params.local_rank],
                output_device=params.local_rank,
                broadcast_buffers=True,
            )

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def cuda(self):
        self.model.cuda()

    def get_scores(self, x, lengths, pred_mask, y, positions=None, langs=None):
        """
        Inputs:
            `x`        : LongTensor of shape (slen, bs)
            `lengths`  : LongTensor of shape (bs,)
        Outputs:
            `sent_emb` : FloatTensor of shape (bs, out_dim)
        With out_dim == emb_dim
        """
        slen, bs = x.size()
        assert lengths.size(0) == bs and lengths.max().item() == slen

        # get transformer last hidden layer
        tensor = self.model(
            "fwd", x=x, lengths=lengths, positions=positions, langs=langs
        )
        assert tensor.size() == (slen, bs, self.out_dim)

        word_scores, loss = self.model(
            "predict", tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True
        )

        return word_scores, loss
