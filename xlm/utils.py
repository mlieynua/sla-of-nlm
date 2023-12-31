# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import getpass
import os
import pickle
import random
import re
import subprocess
import sys

import numpy as np
import torch
from argparse import Namespace

from .logger import create_logger

DUMP_PATH = "/checkpoint/%s/dumped" % getpass.getuser()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def initialize_exp(params: Namespace):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith("--"):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match("^[a-zA-Z0-9_]+$", x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = " ".join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"),
        rank=getattr(params, "global_rank", 0),
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_dump_path(params: Namespace):
    """
    Create a directory to store the experiment.
    """
    dump_path = DUMP_PATH if params.dump_path == "" else params.dump_path
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == "":
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(10))
            if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                break
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    return [None if x is None else x.cuda() for x in args]


def concat_batches(
    x1, len1, lang1_id, x2, len2, lang2_id, pad_idx, eos_idx, reset_positions
):
    """
    Concat batches with different languages.
    """
    assert reset_positions is False or lang1_id != lang2_id
    lengths = len1 + len2
    if not reset_positions:
        lengths -= 1
    slen, bs = lengths.max().item(), lengths.size(0)

    x = x1.new(slen, bs).fill_(pad_idx)
    x[: len1.max().item()].copy_(x1)
    positions = torch.arange(slen)[:, None].repeat(1, bs).to(x1.device)
    langs = x1.new(slen, bs).fill_(lang1_id)

    for i in range(bs):
        l1 = len1[i] if reset_positions else len1[i] - 1
        x[l1 : l1 + len2[i], i].copy_(x2[: len2[i], i])
        if reset_positions:
            positions[l1:, i] -= len1[i]
        langs[l1:, i] = lang2_id

    assert (x == eos_idx).long().sum().item() == (4 if reset_positions else 3) * bs

    return x, lengths, positions, langs


def shuf_order(langs, params=None, n=5):
    """
    Randomize training order.
    """
    if len(langs) == 0:
        return []

    if params is None:
        return [langs[i] for i in np.random.permutation(len(langs))]

    # sample monolingual and parallel languages separately
    mono = [l1 for l1, l2 in langs if l2 is None]
    para = [(l1, l2) for l1, l2 in langs if l2 is not None]

    # uniform / weighted sampling
    p_mono = None
    p_para = None

    s_mono = (
        [
            mono[i]
            for i in np.random.choice(
                len(mono), size=min(n, len(mono)), p=p_mono, replace=True
            )
        ]
        if len(mono) > 0
        else []
    )
    s_para = (
        [
            para[i]
            for i in np.random.choice(
                len(para), size=min(n, len(para)), p=p_para, replace=True
            )
        ]
        if len(para) > 0
        else []
    )

    assert len(s_mono) + len(s_para) > 0
    return [(lang, None) for lang in s_mono] + s_para


def find_modules(module, module_name, module_instance, found):
    """
    Recursively find all instances of a specific module inside a module.
    """
    if isinstance(module, module_instance):
        found.append((module_name, module))
    else:
        for name, child in module.named_children():
            name = ("%s[%s]" if name.isdigit() else "%s.%s") % (module_name, name)
            find_modules(child, name, module_instance, found)
