# -*- coding:utf8 -*-
from torchtext.data import Iterator, BucketIterator
from torchtext import data
import torch


def load_iters(batch_size=32, device="cpu", data_path='data', vectors=None, use_tree=False, limit=None):
    if not use_tree:
        TEXT = data.Field(batch_first=True, include_lengths=True, lower=True)
        LABEL = data.LabelField(batch_first=True)
        TREE = None

        fields = {'premise': ('premise', TEXT),
                  'hypothesis': ('hypothesis', TEXT),
                  'label': ('label', LABEL)}
    else:
        TEXT = data.Field(batch_first=True,
                          lower=True,
                          preprocessing=lambda parse: [t for t in parse if t not in ('(', ')')],
                          include_lengths=True)
        LABEL = data.LabelField(batch_first=True)
        TREE = data.Field(preprocessing=lambda parse: ['reduce' if t == ')' else 'shift' for t in parse if t != '('],
                          batch_first=True)

        TREE.build_vocab([['reduce'], ['shift']])

        fields = {'sentence1_binary_parse': [('premise', TEXT),
                                             ('premise_transitions', TREE)],
                  'sentence2_binary_parse': [('hypothesis', TEXT),
                                             ('hypothesis_transitions', TREE)],
                  'gold_label': ('label', LABEL)}

    train_data, test_data = data.TabularDataset.splits(
        path=data_path,
        train='train.jsonl',
        test='test.jsonl',
        format='json',
        fields=fields,
        filter_pred=lambda ex: ex.label != '-'  # filter the example which label is '-'(means unlabeled)
    )

    if limit is not None and limit < len(train_data.examples):
        train_data = data.Dataset(train_data.examples[:limit], train_data.fields)

    if vectors is not None:
        TEXT.build_vocab(train_data, vectors=vectors, unk_init=torch.Tensor.normal_)
    else:
        TEXT.build_vocab(train_data, max_size=50000)

    LABEL.build_vocab(train_data.label)

    train_data, dev_data = train_data.split([0.9, 0.1])

    train_iter, dev_iter = BucketIterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.premise) + len(x.hypothesis),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )

    test_iter = Iterator(test_data,
                         batch_size=batch_size,
                         device=device,
                         sort=False,
                         sort_within_batch=False,
                         repeat=False,
                         shuffle=False)

    return train_iter, dev_iter, test_iter, TEXT, LABEL, TREE

