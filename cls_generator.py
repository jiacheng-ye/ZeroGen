# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import os
from functools import partial
from typing import List, Optional, Dict, Any, Union
import torch
import torch.nn as nn
import wandb
import numpy as np
from torch.utils.data import SequentialSampler, BatchSampler, DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding
from datasets import Dataset

from tasks import Processor
from utils import save_jsonl

PLACEHOLDER_C = "<C>"
PLACEHOLDER_X = "<X>"
C_KEY = 'C'
X_KEY = 'X'
Y_KEY = 'Y'


class DataGenerator:
    """
    This class represents a generative language model which can be used to generate datasets from instructions.
    """

    def __init__(self, output_dir, task_spec: Dict[str, Any], model: Union['str', 'ModelWrapper'] = None,
                 max_length: int = 40, decay_constant: float = 100,
                 processor: Processor = None, min_length: int = 1,
                 is_stage_two: bool = False, **kwargs):
        self.output_dir = output_dir
        self.model = model
        self.task_name = task_spec["task_name"]
        self.max_length = max_length
        self.min_length = min_length
        self.generate_params = kwargs
        self.is_stage_two = is_stage_two

        self.labels = list(task_spec['labels'].keys())
        self.instructions = {label: task_spec['labels'][label]['instruction'] for label in self.labels}

        self.decay_constant = decay_constant
        if self.decay_constant == 0:  # don't use self-dedbias, so the counter labels can be ignored
            self.counter_labels = {label: [] for label in self.labels}
        else:
            self.counter_labels = {label: task_spec['labels'][label].get('counter_labels', []) for label in self.labels}

        self.processor = processor

    def zero_shot_inference(self, dataset, batch_size: int = 16):
        sentence1_key = self.processor.sentence1_key
        sentence2_key = self.processor.sentence2_key
        instructions = self.instructions
        model = self.model._model
        tokenizer = self.model._tokenizer

        def preprocess_function(examples, label):
            if sentence2_key is None:
                examples = [build_instruction(instructions[label], '', x.replace('<br />', '\n'))
                            for x in examples[sentence1_key]]
            else:
                examples = [build_instruction(instructions[label], c.replace('<br />', '\n'),
                                                   x.replace('<br />', '\n'))
                            for c, x in zip(examples[sentence1_key], examples[sentence2_key])]
            return tokenizer(examples, truncation=True, max_length=512)

        datasets = []
        for i in range(self.processor.num_labels):
            datasets.append(dataset.map(partial(preprocess_function, label=str(i)),
                                        batched=True,
                                        load_from_cache_file=False,
                                        remove_columns=dataset.column_names))

        def lm_loss(dataset):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
            loss_list = []
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(tqdm(dataloader)):
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = batch["input_ids"][..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss(reduce=False, ignore_index=tokenizer.pad_token_id)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
                        shift_labels.size())
                    avg_loss = loss.sum(-1) / (loss > 0).sum(-1)
                    loss_list += avg_loss.tolist()
            return loss_list

        lm_loss_list = np.array([lm_loss(dataset) for dataset in datasets])
        preds = lm_loss_list.argmin(axis=0)
        gold = np.array(dataset['label'])
        expanded_gold = np.expand_dims(gold, axis=0)
        lm_loss_exp = np.exp(lm_loss_list)
        gold_probs = np.take_along_axis(lm_loss_exp, expanded_gold, axis=0)/lm_loss_exp.sum(axis=0)
        acc = (preds == gold).sum() / len(preds)
        logging.info("Zero-shot accuracy is " + str(acc))
        return gold_probs

    def generate_dataset(self, input_texts: Optional[List[str]], num_entries_per_input: Optional[int] = None,
                         batch_size: int = 16, log_every: int = 10000) -> List[Dict]:
        generate_with_inputs = input_texts is not None

        if generate_with_inputs:
            num_instructions = batch_size // num_entries_per_input
        else:
            input_texts = list(range(num_entries_per_input))
            num_entries_per_input = 1
            num_instructions = batch_size

        sampler = BatchSampler(SequentialSampler(input_texts), batch_size=num_instructions, drop_last=False)
        dataset = []
        new_dataset = []
        log_count = 1
        for i, indices in enumerate(tqdm(sampler)):
            to_add = []
            input_texts_or_ids = [input_texts[i] for i in indices]
            for label in self.labels:
                outputs = self._generate_dataset_entries(input_texts_or_ids, label=label,
                                                         num_samples=num_entries_per_input,
                                                         generate_with_inputs=generate_with_inputs)

                to_add += outputs

            to_add = postprocess_dataset(to_add, generate_with_inputs)

            new_dataset += to_add

            overall_size = len(dataset) + len(new_dataset)
            if self.processor and overall_size >= log_count * log_every and self.is_stage_two:
                res_dict = {}

                # combine the new dataset with old dataset
                dataset += new_dataset
                table = wandb.Table(data=[[ex[C_KEY], ex[X_KEY], ex[Y_KEY]] for ex in new_dataset[:100]],
                                    columns=[C_KEY, X_KEY, Y_KEY])
                res_dict.update({'#Train': len(dataset), "examples": table})

                # re-init model and fine-tune from scratch
                self.processor.load_model()  # use the initial model

                # train the model with full dataset
                hf_dataset = convert_to_hf_dataset(dataset,
                                                   sentence1_key=self.processor.sentence1_key,
                                                   sentence2_key=self.processor.sentence2_key)
                self.processor.train(*self.processor.load_train_val(hf_dataset))
                logging.info(f"Test results using {len(dataset)} training data: ")

                # check the metric on validation dataset with new model
                val_metric = self.processor.validate()
                res_dict.update({"val": val_metric})

                logging.info(res_dict)
                wandb.log(res_dict)

                log_count += 1
                new_dataset = []

                logging.info("Save to disk...")
                dataset_path = os.path.join(self.output_dir, f'{self.task_name}-dataset.jsonl')
                save_jsonl(dataset, dataset_path)
        dataset += new_dataset
        return dataset

    def _generate_dataset_entries(self, input_texts_or_ids: Union[List[str], List[int]], label: str, num_samples: int,
                                  generate_with_inputs: bool) -> List[Dict]:
        instructions = [build_instruction(self.instructions[label], input_text_or_id)
                        for input_text_or_id in input_texts_or_ids]
        counter_instructions = []
        for other_label in self.counter_labels[label]:
            counter_instructions += [build_instruction(self.instructions[other_label], input_text_or_id)
                                     for input_text_or_id in input_texts_or_ids]

        model_outputs = self.model.generate_self_debiasing(
            input_texts=instructions,
            debiasing_texts=counter_instructions,
            num_samples=num_samples,
            decay_constant=self.decay_constant,
            min_length=self.max_length,
            max_length=self.max_length,
            label=label,
            **self.generate_params
        )
        outputs = []
        for i, input_text_or_id in enumerate(input_texts_or_ids):
            for j in range(num_samples):
                output = process_output(input_text=input_text_or_id,
                                        output_text=model_outputs[i * num_samples + j],
                                        label=label, generate_with_inputs=generate_with_inputs,
                                        min_length=self.min_length, task_name=self.task_name)
                if output is not None:
                    outputs.append(output)

        return outputs


def convert_to_hf_dataset(entries: List[Dict], sentence1_key: str, sentence2_key: Optional[str]) -> Dataset:
    res = {sentence1_key: [], 'label': [], 'idx': []}
    if sentence2_key is not None:
        res.update({sentence2_key: []})
    for i, entry in enumerate(entries):
        res['label'].append(entry[Y_KEY])
        res['idx'].append(i)
        if sentence2_key is not None:
            res[sentence1_key].append(entry[C_KEY])
            res[sentence2_key].append(entry[X_KEY])
        else:
            res[sentence1_key].append(entry[X_KEY])
    return Dataset.from_dict(res)


def build_instruction(instruction: str, c: Union[str, int], x: Optional[str] = None) -> str:
    if isinstance(c, int):
        return instruction
    output = instruction.replace(PLACEHOLDER_C, c)
    if x:
        output = output.replace(PLACEHOLDER_X, x)
    return output


def process_output(input_text: Union[str, int], output_text: str, label: str, generate_with_inputs: bool,
                   min_length: int, task_name: str) -> Optional[Dict]:
    if task_name == "qnli":
        if '?' in output_text:
            output_text = output_text.split('?')[0] + "?"
        else:
            return None
    elif '"' in output_text:
        output_text = output_text.split('"')[0]
    elif '\n' in output_text:
        output_text = output_text.split('\n')[0]
    elif '.' in output_text:
        sentences = output_text.split('.')
        output_text = '.'.join(sentences[:-1]) + '.'
    else:
        return None

    if len(output_text.strip().split(' ')) >= min_length:
        if generate_with_inputs:
            c = input_text
            x = output_text
        else:
            c = output_text
            x = None
        return {C_KEY: c, X_KEY: x, Y_KEY: float(label) if task_name == "stsb" else int(label)}
    return None


def postprocess_dataset(dataset: List[Dict], generate_with_inputs: bool) -> List[Dict]:
    postprocessed_dataset = []
    for example in dataset:
        if generate_with_inputs:  # force the generated x to be different from c
            if example[C_KEY] == example[X_KEY]:
                continue

        postprocessed_dataset.append(json.dumps(example))
    postprocessed_dataset = [json.loads(i) for i in list(dict.fromkeys(postprocessed_dataset))]
    return postprocessed_dataset

