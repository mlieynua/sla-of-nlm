# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from argparse import Namespace
from logging import getLogger

import torch

from .transformer import TransformerModel
from xlm.data.dictionary import Dictionary

logger = getLogger()


def check_model_params(params: Namespace) -> None:
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    s = params.word_mask_keep_rand.split(",")
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # reload a pretrained model
    if params.reload_model != "":
        assert os.path.isfile(params.reload_model)


def build_model(params: None, dico: Dictionary) -> TransformerModel:
    """
    Build model.
    """
    # build
    model = TransformerModel(params, dico)

    # reload a pretrained model
    if params.reload_model != "":
        logger.info("Reloading model from %s ..." % params.reload_model)
        reloaded = torch.load(
            params.reload_model,
            map_location=lambda storage, loc: storage.cuda(params.local_rank),
        )["model"]
        if all([k.startswith("module.") for k in reloaded.keys()]):
            reloaded = {k[len("module.") :]: v for k, v in reloaded.items()}

        model.load_state_dict(reloaded, strict=False)

    logger.info("Model: {}".format(model))
    logger.info(
        "Number of parameters (model): %i"
        % sum([p.numel() for p in model.parameters() if p.requires_grad])
    )

    return model.cuda()
