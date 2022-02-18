#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import logging
import math
import os

import torch
import wandb
from datasets import Dataset, DatasetDict, load_from_disk, load_metric, load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, \
    DataCollatorWithPadding, EvalPrediction, AutoConfig, EarlyStoppingCallback, IntervalStrategy


from tasks.base_processor import Processor
from tasks.qa_utils import postprocess_qa_predictions, QuestionAnsweringTrainer
from utils import set_seed


class QAProcessor(Processor):
    def __init__(self, task_name, model_name, model_ckpt, output_dir, device, **train_args):
        super().__init__(task_name, model_name, model_ckpt, output_dir, device, **train_args)
        self.train_key = "train"
        self.validation_key = "validation"

        self.load_model()
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        self.pad_on_right = self.tokenizer.padding_side == "right"
        self.max_length = 384  # The maximum length of a feature (question and context)
        self.doc_stride = 128  # The authorized overlap between two part of the context when splitting it is needed.
        self.load_dataset()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        load_name = self.model_ckpt if self.model_ckpt is not None else self.model_name
        self.model = AutoModelForQuestionAnswering.from_pretrained(load_name).to(self.device)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    def load_dataset(self):
        data_path = f'data/{self.task_name}'
        if os.path.exists(data_path):
            self.dataset = load_from_disk(data_path)
        else:
            self.dataset = load_dataset(self.task_name)
            self.dataset.save_to_disk(data_path)

        self.encoded_dataset = DatasetDict()
        self.encoded_dataset[self.train_key] = self.dataset[self.train_key].map(
            self._prepare_train_features,
            batched=True,
            load_from_cache_file=False,
            remove_columns=self.dataset[self.train_key].column_names)
        self.encoded_dataset[self.validation_key] = self.dataset[self.validation_key].map(
            self._prepare_validation_features,
            batched=True,
            load_from_cache_file=False,
            remove_columns=self.dataset[self.validation_key].column_names)
        self.metric = load_metric('squad')
        self.main_metric_name = 'eval_f1'

    def load_train_val(self, dataset, seed):
        if dataset is None:
            dataset = self.dataset[self.train_key]
        elif isinstance(dataset, Dataset):
            dataset = dataset
        else:
            raise RuntimeError()
        dataset = dataset.train_test_split(test_size=0.1, seed=seed)
        train_dataset, val_dataset = dataset['train'], dataset['test']
        encoded_train_dataset = train_dataset.map(
            self._prepare_train_features,
            batched=True,
            load_from_cache_file=False,
            remove_columns=train_dataset.column_names)
        encoded_val_dataset = val_dataset.map(
            self._prepare_validation_features,
            batched=True,
            load_from_cache_file=False,
            remove_columns=val_dataset.column_names)
        return encoded_train_dataset, encoded_val_dataset, val_dataset

    def _prepare_train_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]
        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question" if self.pad_on_right else "context"],
            examples["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        return tokenized_examples

    def _prepare_validation_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]
        tokenized_examples = self.tokenizer(
            examples["question" if self.pad_on_right else "context"],
            examples["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def _post_processing_function(self, examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=self.task_name == "squad_v2",
            n_best_size=20,
            max_answer_length=30,
            null_score_diff_threshold=0.0,
            output_dir=self.output_dir,
            log_level=logging.INFO,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if self.task_name == "squad_v2":
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

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
            load_best_model_at_end=True,
            warmup_steps=1000
        )

        def compute_metrics(p: EvalPrediction):
            return self.metric.compute(predictions=p.predictions, references=p.label_ids)

        self.trainer = QuestionAnsweringTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_examples=eval_examples,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            post_process_function=self._post_processing_function,
            compute_metrics=compute_metrics,
        )

        self.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

        if train:
            self.trainer.train()

    def validate(self, val_dataset=None, show_error_cases=False, show_tsne=False, return_probs=False):
        if val_dataset is None:
            encoded_val_dataset = self.encoded_dataset[self.validation_key]  # by default we test on the validation set
            val_dataset = self.dataset[self.validation_key]
        else:
            encoded_val_dataset = val_dataset.map(
                    self._prepare_validation_features,
                    batched=True,
                    load_from_cache_file=False,
                    remove_columns=val_dataset.column_names)
        eval_metric = self.trainer.evaluate(encoded_val_dataset, val_dataset)
        return eval_metric