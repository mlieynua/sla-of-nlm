# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from argparse import ArgumentParser, Namespace
import json
import random

import numpy as np
import torch

from xlm.data.loader import check_data_params, load_data
from xlm.evaluation.evaluator import SingleEvaluator
from xlm.model import build_model, check_model_params
from xlm.slurm import init_distributed_mode
from xlm.trainer import SingleTrainer
from xlm.utils import initialize_exp, shuf_order


def parse_args() -> Namespace:
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = ArgumentParser(description="Language transfer")

    # fmt: off
    # main parameters
    parser.add_argument("--dump_path", type=str, default="/cl/work4/miyu-ob/seed_dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="exp00",
                        help="Experiment ID")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--save_epoch", type=str, default="99",
                        help="Epoch which the model is saved on.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool, default=True,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=2,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=256,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=12,
                        help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0.1,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool, default=True,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--use_lang_emb", type=bool, default=True,
                        help="Use language embedding")

    # masked language modeling task parameters
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="Fraction of words for which we need to make a prediction")
    parser.add_argument("--word_mask_keep_rand", type=str, default="0.8,0.1,0.1",
                        help="Fraction of words to mask out / keep / randomize, among the words to predict")

    # data
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--lgs", type=str, default="",
                        help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--max_len", type=int, default=200,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--group_by_size", type=bool, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")

    # training parameters
    parser.add_argument("--optimizer", type=str, default="adam_inverse_sqrt,lr=0.00020,warmup_updates=30000,beta1=0.9,beta2=0.999,weight_decay=0.01,eps=0.000001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=1,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=4,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")

    # training steps
    parser.add_argument("--mlm_steps", type=str, default="",
                        help="Masked prediction steps (MLM / TLM)")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")
    # fmt: on

    return parser.parse_args()


def torch_fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def main(params: Namespace):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # initialize the experiment
    logger = initialize_exp(params)

    # load data
    data = load_data(params)

    # build model
    model = build_model(params, data["dico"])

    # build trainer, reload potential checkpoints / build evaluator
    trainer = SingleTrainer(model, data, params)
    evaluator = SingleEvaluator(trainer, data, params)

    # language model training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < trainer.epoch_size:

            # MLM steps (also includes TLM if lang2 is not None)
            if len(params.mlm_steps) == 1:
                for lang1, lang2 in shuf_order(params.mlm_steps, params):
                    trainer.mlm_step(lang1, lang2)
            elif len(params.mlm_steps) == 2:
                assert params.mlm_steps[1] == ("en", None)
                if trainer.epoch % 2 == 0:
                    lang1, lang2 = shuf_order(params.mlm_steps, params)[1]
                    trainer.mlm_step(lang1, lang2)
                else:
                    lang1, lang2 = shuf_order(params.mlm_steps, params)[0]
                    trainer.mlm_step(lang1, lang2)

            trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        scores = evaluator.run_all_evals(trainer)

        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)


if __name__ == "__main__":
    params = parse_args()

    torch_fix_seed(params.random_seed)

    # check parameters
    check_data_params(params)
    check_model_params(params)

    # run experiment
    main(params)
