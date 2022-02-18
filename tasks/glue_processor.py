#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Union

import numpy as np
import os
import torch
from datasets import load_metric, Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, \
    IntervalStrategy, DataCollatorWithPadding, EarlyStoppingCallback

from tasks.base_processor import Processor

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli_mismatched": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("sentence", "question"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2")
}


class GLUEProcessor(Processor):
    def __init__(self, task_name, model_name, model_ckpt, output_dir, device, **train_args):
        super().__init__(task_name, model_name, model_ckpt, output_dir, device, **train_args)
        self.train_key = "train"
        self.validation_key = "validation"
        self.test_key = "test"

        self.load_model()
        self.load_dataset()

    def load_model(self):
        self.num_labels = 3 if self.task_name.startswith("mnli") else 1 if self.task_name == "stsb" else 2
        self.is_regression = True if self.task_name == 'stsb' else False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        load_name = self.model_ckpt if self.model_ckpt is not None else self.model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(load_name, num_labels=self.num_labels).to(self.device)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    def load_dataset(self):
        if self.task_name == 'mnli_mismatched':
            self.validation_key = "validation_mismatched"
            self.test_key = "test_mismatched"
            self.task_name = 'mnli'
        elif self.task_name == 'mnli_matched':
            self.validation_key = "validation_matched"
            self.test_key = "test_matched"
            self.task_name = 'mnli'

        data_path = f'data/{self.task_name}'
        if os.path.exists(data_path):
            self.dataset = load_from_disk(data_path)
        else:
            self.dataset = load_dataset('glue', self.task_name)
            self.dataset.save_to_disk(data_path)

        self.sentence1_key, self.sentence2_key = task_to_keys[self.task_name]
        self.encoded_dataset = self._encode_dataset(self.dataset)
        # no label for test set on GLUE tasks
        self.encoded_dataset[self.test_key] = self.encoded_dataset[self.test_key].remove_columns("label")
        self.metric = load_metric("glue", self.task_name)
        self.main_metric_name = "eval_spearman" if self.task_name == "stsb" \
            else "eval_matthews_correlation" if self.task_name == "cola" else "eval_accuracy"

    def _encode_dataset(self, dataset: Union[Dataset, DatasetDict]):
        remove_columns = [col for col in self.dataset[self.train_key].column_names if col != 'label']
        encoded_dataset = dataset.map(self._preprocess_function, batched=True,
                                      load_from_cache_file=False,
                                      remove_columns=remove_columns)
        return encoded_dataset

    def _preprocess_function(self, examples):
        if self.sentence2_key is None:
            return self.tokenizer(examples[self.sentence1_key], truncation=True, max_length=128)
        return self.tokenizer(examples[self.sentence1_key], examples[self.sentence2_key], truncation=True,
                              max_length=128)

    def _compute_metrics(self, eval_pred, metric_key_prefix='eval'):
        predictions, labels = eval_pred
        if self.is_regression:
            predictions = predictions[:, 0]
        else:
            predictions = np.argmax(predictions, axis=1)
        metrics = self.metric.compute(predictions=predictions, references=labels)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return metrics

    def load_train_val(self, dataset, seed=42):
        if dataset is None:  # use standard training set by default
            encoded_dataset = self.encoded_dataset[self.train_key]
        elif isinstance(dataset, Dataset):
            encoded_dataset = self._encode_dataset(dataset)
        else:
            raise RuntimeError()
        encoded_dataset = encoded_dataset.train_test_split(test_size=0.1, seed=seed)
        train_samples, val_samples = encoded_dataset['train'], encoded_dataset['test']
        return train_samples, val_samples

    def train(self, train_dataset, eval_dataset=None, eval_examples=None, train=True):

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.train_args['num_epochs'],
            per_device_train_batch_size=self.train_args['train_batch_size'],
            per_device_eval_batch_size=self.train_args['train_batch_size'],
            learning_rate=self.train_args['learning_rate'],
            weight_decay=0.01,
            evaluation_strategy=IntervalStrategy.EPOCH,
            metric_for_best_model=self.main_metric_name,
            save_strategy=IntervalStrategy.EPOCH,
            save_total_limit=10,
            load_best_model_at_end=True
        )

        self.trainer = Trainer(
            model=self.model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics, data_collator=self.data_collator
        )

        self.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

        if train:
            self.trainer.train()

    def validate(self, val_dataset=None):
        if val_dataset is None:  # by default we use the standard validation set
            encoded_val_dataset = self.encoded_dataset[self.validation_key]
        elif isinstance(val_dataset, Dataset):
            encoded_val_dataset = self._encode_dataset(val_dataset)
        else:
            raise RuntimeError()

        batch_size = self.train_args['train_batch_size']
        eval_dataloader = DataLoader(encoded_val_dataset,
                                     batch_size=batch_size,
                                     collate_fn=self.data_collator)

        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch, output_hidden_states=True)
                predictions = outputs.logits if self.is_regression else outputs.logits.argmax(dim=-1)

                self.metric.add_batch(
                    predictions=predictions,
                    references=batch["labels"],
                )

        eval_metric = self.metric.compute()
        # Prefix all keys with eval + '_'
        for key in list(eval_metric.keys()):
            if not key.startswith("eval_"):
                eval_metric[f"eval_{key}"] = eval_metric.pop(key)

        return eval_metric