#!/usr/bin/python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class Processor(ABC):
    def __init__(self, task_name, model_name, model_ckpt, output_dir, device, **train_args):
        self.task_name = task_name
        self.model_name = model_name
        self.model_ckpt = model_ckpt
        self.output_dir = output_dir
        self.device = device
        self.train_args = train_args

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def load_train_val(self, dataset, seed):
        pass

    @abstractmethod
    def train(self, train_dataset, eval_dataset, eval_examples):
        pass

    @abstractmethod
    def validate(self, val_dataset, **kwargs):
        pass