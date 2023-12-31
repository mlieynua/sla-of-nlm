# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from logging import getLogger
from argparse import Namespace

import numpy as np
import torch
from typing import Iterator

logger = getLogger()


class StreamDataset(object):
    def __init__(self, sent: np.ndarray, pos: np.ndarray, bs: int, params: Namespace):
        """
        Prepare batches for data iterator.
        """
        bptt = params.bptt
        self.eos = params.eos_index

        # checks
        assert len(pos) == (sent == self.eos).sum()
        assert len(pos) == (sent[pos[:, 1]] == self.eos).sum()

        n_tokens = len(sent)
        n_batches = math.ceil(n_tokens / (bs * bptt))
        t_size = n_batches * bptt * bs

        buffer = np.zeros(t_size, dtype=sent.dtype) + self.eos
        buffer[t_size - n_tokens :] = sent
        buffer = buffer.reshape((bs, n_batches * bptt)).T
        self.data = np.zeros((n_batches * bptt + 1, bs), dtype=sent.dtype) + self.eos
        self.data[1:] = buffer

        self.bptt = bptt
        self.n_tokens = n_tokens
        self.n_batches = n_batches
        self.n_sentences = len(pos)
        self.lengths = torch.LongTensor(bs).fill_(bptt)

    def __len__(self) -> int:
        """
        Number of sentences in the dataset.
        """
        return self.n_sentences

    def get_iterator(self, shuffle: bool, subsample: int=1) -> Iterator[torch.Tensor]:
        """
        Return a sentences iterator.
        """
        indexes = (np.random.permutation if shuffle else range)(
            self.n_batches // subsample
        )
        for i in indexes:
            a = self.bptt * i
            b = self.bptt * (i + 1)
            yield torch.from_numpy(self.data[a:b].astype(np.int64)), self.lengths


class Dataset(object):
    def __init__(self, sent, pos, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size

        self.sent = sent
        self.pos = pos
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # check number of sentences
        assert len(self.pos) == (self.sent == self.eos_index).sum()

        # sanity checks
        self.check()

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos) == (self.sent[self.pos[:, 1]] == eos).sum()

    def batch_sentences(self, sentences):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(
            self.pad_index
        )

        sent[0] = self.eos_index
        for i, s in enumerate(sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1 : lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def get_batches_iterator(self, batches):
        """
        Return a sentences iterator, given the associated sentence batches.
        """

        for sentence_ids in batches:
            # pos: [[ 0 17]]
            pos = self.pos[sentence_ids]
            # sent: [array([15802,  9071,  6420,   903,  2227,  1559,  1366,  8798,  1113, 24050,  9071,  9197,  3168,  1342,  2782, 24082,    74], dtype=uint16)]
            sent = [self.sent[a:b] for a, b in pos]
            # sent: (tensor([[    1], [15802], [ 9071], [ 6420], [  903], [ 2227], [ 1559], [ 1366], [ 8798], [ 1113], [24050], [ 9071], [ 9197], [ 3168], [ 1342], [ 2782], [24082], [   74], [    1]]),
            # tensor([19]))
            sent = self.batch_sentences(sent)
            yield sent

    def get_iterator(
        self,
    ):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos)
        assert 0 < n_sentences <= len(self.pos)

        # sentence lengths
        lengths = self.lengths + 2

        # select sentences to iterate over
        indices = np.arange(n_sentences)

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        batches = np.array_split(
            indices, math.ceil(len(indices) * 1.0 / self.batch_size)
        )

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches)


class ParallelDataset(Dataset):
    def __init__(self, sent1: np.ndarray, pos1: np.ndarray, sent2: np.ndarray, pos2: np.ndarray, params: Namespace):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size

        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        # check number of sentences
        assert len(self.pos1) == (self.sent1 == self.eos_index).sum()
        assert len(self.pos2) == (self.sent2 == self.eos_index).sum()

        # remove empty sentences
        self.remove_empty_sentences()

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)

    def check(self) -> None:
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos1) == len(self.pos2) > 0  # check number of sentences
        assert (
            len(self.pos1) == (self.sent1[self.pos1[:, 1]] == eos).sum()
        )  # check sentences indices
        assert (
            len(self.pos2) == (self.sent2[self.pos2[:, 1]] == eos).sum()
        )  # check sentences indices
        assert eos <= self.sent1.min() < self.sent1.max()  # check dictionary indices
        assert eos <= self.sent2.min() < self.sent2.max()  # check dictionary indices
        assert self.lengths1.min() > 0  # check empty sentences
        assert self.lengths2.min() > 0  # check empty sentences

    def remove_empty_sentences(self) -> None:
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        indices = indices[self.lengths2[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len: int) -> None:
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            pos1 = self.pos1[sentence_ids]
            pos2 = self.pos2[sentence_ids]
            sent1 = self.batch_sentences([self.sent1[a:b] for a, b in pos1])
            sent2 = self.batch_sentences([self.sent2[a:b] for a, b in pos2])
            yield (sent1, sent2, sentence_ids) if return_indices else (sent1, sent2)

    def get_iterator(
        self, shuffle, group_by_size=False, n_sentences=-1, return_indices=False
    ):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # sentence lengths
        lengths = self.lengths1 + self.lengths2 + 4

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind="mergesort")]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        batches = np.array_split(
            indices, math.ceil(len(indices) * 1.0 / self.batch_size)
        )

        # optionally shuffle batches
        if shuffle:
            np.random.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)
