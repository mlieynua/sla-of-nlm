# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
from argparse import Namespace

import numpy as np
import torch
from typing import Dict, Union, Tuple

from .dataset import ParallelDataset, StreamDataset
from .dictionary import BOS_WORD, EOS_WORD, MASK_WORD, PAD_WORD, UNK_WORD, Dictionary

logger = getLogger()
type_dataset = Dict[
    str,
    Dict[Union[str, Tuple[str, str]], Dict[str, Union[StreamDataset, ParallelDataset]]],
]
type_mono_dataset = Dict[str, Dict[str, Dict[str, StreamDataset]]]
type_para_dataset = Dict[str, Dict[Tuple[str, str], Dict[str, ParallelDataset]]]
type_data = Dict[str, Union[Dictionary, np.ndarray, Dict[str, int]]]


def process_binarized(data: type_data, params: Namespace) -> type_dataset:
    """
    Process a binarized dataset and log main statistics.
    """
    dico = data["dico"]
    assert (
        (data["sentences"].dtype == np.uint16)
        and (len(dico) < 1 << 16)
        or (data["sentences"].dtype == np.int32)
        and (1 << 16 <= len(dico) < 1 << 31)
    )
    logger.info(
        "%i words (%i unique) in %i sentences. %i unknown words (%i unique) covering %.2f%% of the data."
        % (
            len(data["sentences"]) - len(data["positions"]),
            len(dico),
            len(data["positions"]),
            sum(data["unk_words"].values()),
            len(data["unk_words"]),
            100.0
            * sum(data["unk_words"].values())
            / (len(data["sentences"]) - len(data["positions"])),
        )
    )
    if params.max_vocab != -1:
        assert params.max_vocab > 0
        logger.info("Selecting %i most frequent words ..." % params.max_vocab)
        dico.max_vocab(params.max_vocab)
        data["sentences"][data["sentences"] >= params.max_vocab] = dico.index(UNK_WORD)
        unk_count = (data["sentences"] == dico.index(UNK_WORD)).sum()
        logger.info(
            "Now %i unknown words covering %.2f%% of the data."
            % (
                unk_count,
                100.0 * unk_count / (len(data["sentences"]) - len(data["positions"])),
            )
        )
    if params.min_count > 0:
        logger.info("Selecting words with >= %i occurrences ..." % params.min_count)
        dico.min_count(params.min_count)
        data["sentences"][data["sentences"] >= len(dico)] = dico.index(UNK_WORD)
        unk_count = (data["sentences"] == dico.index(UNK_WORD)).sum()
        logger.info(
            "Now %i unknown words covering %.2f%% of the data."
            % (
                unk_count,
                100.0 * unk_count / (len(data["sentences"]) - len(data["positions"])),
            )
        )
    if (data["sentences"].dtype == np.int32) and (len(dico) < 1 << 16):
        logger.info("Less than 65536 words. Moving data from int32 to uint16 ...")
        data["sentences"] = data["sentences"].astype(np.uint16)
    return data


def load_binarized(path: str, params: Namespace) -> type_dataset:
    """
    Load a binarized dataset.
    """
    assert path.endswith(".pth")
    if getattr(params, "multi_gpu", False):
        split_path = "%s.%i.pth" % (path[:-4], params.local_rank)
        if os.path.isfile(split_path):
            path = split_path
    assert os.path.isfile(path), path
    logger.info("Loading data from %s ..." % path)
    data = torch.load(path)
    data = process_binarized(data, params)
    return data


def set_dico_parameters(
    params: Namespace,
    data: Dict[str, Dict[str, Dict[str, StreamDataset]]],
    dico: Dictionary,
) -> None:
    """
    Update dictionary parameters.
    """
    if "dico" in data:
        assert data["dico"] == dico
    else:
        data["dico"] = dico

    n_words = len(dico)
    bos_index = dico.index(BOS_WORD)
    eos_index = dico.index(EOS_WORD)
    pad_index = dico.index(PAD_WORD)
    unk_index = dico.index(UNK_WORD)
    mask_index = dico.index(MASK_WORD)
    if hasattr(params, "bos_index"):
        assert params.n_words == n_words
        assert params.bos_index == bos_index
        assert params.eos_index == eos_index
        assert params.pad_index == pad_index
        assert params.unk_index == unk_index
        assert params.mask_index == mask_index
    else:
        params.n_words = n_words
        params.bos_index = bos_index
        params.eos_index = eos_index
        params.pad_index = pad_index
        params.unk_index = unk_index
        params.mask_index = mask_index


def load_mono_data(params: Namespace, data: type_mono_dataset) -> None:
    """
    Load monolingual data.
    """
    data["mono"] = {}
    data["mono_stream"] = {}

    for lang in params.mono_dataset.keys():

        logger.info("============ Monolingual data (%s)" % lang)

        assert lang in params.langs and lang not in data["mono"]
        data["mono_stream"][lang] = {}

        for splt in ["train", "valid", "test"]:

            # load data / update dictionary parameters / update data
            mono_data = load_binarized(params.mono_dataset[lang][splt], params)
            set_dico_parameters(params, data, mono_data["dico"])

            # create stream dataset
            bs = params.batch_size if splt == "train" else 1
            data["mono_stream"][lang][splt] = StreamDataset(
                mono_data["sentences"], mono_data["positions"], bs, params
            )

            logger.info("")

    logger.info("")


def load_para_data(params: Namespace, data: type_para_dataset) -> None:
    """
    Load parallel data.
    """
    data["para"] = {}

    for src, tgt in params.para_dataset.keys():

        logger.info("============ Parallel data (%s-%s)" % (src, tgt))

        assert (src, tgt) not in data["para"]
        data["para"][(src, tgt)] = {}

        for splt in ["train", "valid", "test"]:

            # load binarized datasets
            src_path, tgt_path = params.para_dataset[(src, tgt)][splt]
            src_data = load_binarized(src_path, params)
            tgt_data = load_binarized(tgt_path, params)

            # update dictionary parameters
            set_dico_parameters(params, data, src_data["dico"])
            set_dico_parameters(params, data, tgt_data["dico"])

            # create ParallelDataset
            dataset = ParallelDataset(
                src_data["sentences"],
                src_data["positions"],
                tgt_data["sentences"],
                tgt_data["positions"],
                params,
            )

            # remove empty and too long sentences
            if splt == "train":
                dataset.remove_empty_sentences()
                dataset.remove_long_sentences(params.max_len)

            data["para"][(src, tgt)][splt] = dataset
            logger.info("")

    logger.info("")


def check_data_params(params: Namespace) -> None:
    """
    Check datasets parameters.
    """
    # data path
    assert os.path.isdir(params.data_path), params.data_path

    # check languages
    params.langs = params.lgs.split("-")
    assert len(params.langs) == len(set(params.langs)) >= 1
    # assert sorted(params.langs) == params.langs
    params.id2lang = {k: v for k, v in enumerate(sorted(params.langs))}
    params.lang2id = {k: v for v, k in params.id2lang.items()}
    params.n_langs = len(params.langs)

    # MLM / TLM steps
    mlm_steps = [s.split("-") for s in params.mlm_steps.split(",") if len(s) > 0]
    params.mlm_steps = [(s[0], None) if len(s) == 1 else tuple(s) for s in mlm_steps]
    assert all(
        [
            (l1 in params.langs) and (l2 in params.langs or l2 is None)
            for l1, l2 in params.mlm_steps
        ]
    )
    assert len(params.mlm_steps) == len(set(params.mlm_steps))

    # check monolingual datasets
    required_mono = set([l1 for l1, l2 in (params.mlm_steps) if l2 is None])
    params.mono_dataset = {
        lang: {
            splt: os.path.join(params.data_path, "%s.%s.pth" % (splt, lang))
            for splt in ["train", "valid", "test"]
        }
        for lang in params.langs
        if lang in required_mono
    }
    for paths in params.mono_dataset.values():
        for p in paths.values():
            if not os.path.isfile(p):
                logger.error(f"{p} not found")
    assert all(
        [
            all([os.path.isfile(p) for p in paths.values()])
            for paths in params.mono_dataset.values()
        ]
    )

    # check parallel datasets
    required_para_train = set(params.mlm_steps)
    params.para_dataset = {
        (src, tgt): {
            splt: (
                os.path.join(
                    params.data_path,
                    "%s.%s.pth" % (splt, src),
                ),
                os.path.join(
                    params.data_path,
                    "%s.%s.pth" % (splt, tgt),
                ),
            )
            for splt in ["train", "valid", "test"]
            if splt != "train"
            or (src, tgt) in required_para_train
            or (tgt, src) in required_para_train
        }
        for src in params.langs
        for tgt in params.langs
        if src < tgt
    }
    for paths in params.para_dataset.values():
        for p1, p2 in paths.values():
            if not os.path.isfile(p1):
                logger.error(f"{p1} not found")
            if not os.path.isfile(p2):
                logger.error(f"{p2} not found")
    assert all(
        [
            all(
                [os.path.isfile(p1) and os.path.isfile(p2) for p1, p2 in paths.values()]
            )
            for paths in params.para_dataset.values()
        ]
    )


def load_data(
    params: Namespace,
) -> type_dataset:
    """
    Load monolingual data.
    The returned dictionary contains:
        - dico (dictionary)
        - vocab (FloatTensor)
        - train / valid / test (monolingual datasets)
    """
    data = {}

    # monolingual datasets
    load_mono_data(params, data)

    # parallel datasets
    load_para_data(params, data)

    # monolingual data summary
    logger.info("============ Data summary")
    for lang, v in data["mono_stream"].items():
        for data_set in v.keys():
            logger.info(
                "{: <18} - {: >5} - {: >12}:{: >10}".format(
                    "Monolingual data", data_set, lang, len(v[data_set])
                )
            )

    # parallel data summary
    for (src, tgt), v in data["para"].items():
        for data_set in v.keys():
            logger.info(
                "{: <18} - {: >5} - {: >12}:{: >10}".format(
                    "Parallel data", data_set, "%s-%s" % (src, tgt), len(v[data_set])
                )
            )

    logger.info("")
    return data
