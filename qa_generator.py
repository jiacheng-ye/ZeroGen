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

import json
import logging
from typing import List, Optional, Dict, Any, Union

import pandas as pd
import spacy
import wandb
from datasets import Dataset
from torch.utils.data import SequentialSampler, BatchSampler
from tqdm import tqdm

from tasks import Processor

PLACEHOLDER_CONTEXT = "<C>"
PLACEHOLDER_ANSWER = "<Y>"
PLACEHOLDER_QUESTION = "<X>"


class QADataGenerator:

    def __init__(self, output_dir, task_spec: Dict[str, Any], model: Union[str, 'ModelWrapper'] = None,
                 max_length: int = 40, min_length: int = 1, processor: Processor = None,
                 seed: int = 42, **kwargs):

        self.model = model
        self.task_name = task_spec["task_name"].split('-')[0]
        self.max_length = max_length
        self.min_length = min_length
        self.generate_params = kwargs

        self.instruction = task_spec['instruction']
        self.processor = processor
        self.seed = seed
        self.output_dir = output_dir

    def zero_shot_inference(self, batch_size: int = 16) -> Dict:
        dataset = self.processor.dataset[self.processor.validation_key]

        sampler = BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False)

        predictions = []
        references = []
        for indices in tqdm(sampler):
            batch = [dataset[i] for i in indices]

            def preprocess_function(example):
                return self.instruction.replace(PLACEHOLDER_CONTEXT, example['context']) \
                    .replace(PLACEHOLDER_QUESTION, example['question'])

            instructions = [preprocess_function(ex) for ex in batch]
            model_outputs = self.model.generate_self_debiasing(
                input_texts=instructions,
                debiasing_texts=[],
                num_samples=1,
                min_length=self.max_length,
                max_length=self.max_length,
                **self.generate_params
            )
            for example, output in zip(batch, model_outputs):
                prediction = {'id': example['id'], 'prediction_text': ''}
                text = postprocess_answer(output_text=output, min_length=self.min_length)
                if text is not None:
                    prediction['prediction_text'] = text
                predictions.append(prediction)

                references.append({'id': example['id'], 'answers': example['answers']})
        metric = self.processor.metric.compute(predictions=predictions, references=references)
        logging.info(f"Zero-shot metric {str(metric)}")
        return metric

    def generate_answer_ner(self) -> Dataset:
        nlp = spacy.load("en_core_web_sm")
        dataset = self.processor.dataset[self.processor.train_key]
        columns = dataset.format['columns']

        def sample_ner(example):
            doc = nlp(example['context'])
            aug_examples = []
            for i, ent in enumerate(doc.ents):
                tmp = example['aug_examples'][0].copy()
                tmp['question'] = ''
                tmp['answers'] = {'answer_start': [ent.start_char], 'text': [ent.text]}
                tmp['id'] = tmp['id']+'-'+str(i)
                aug_examples.append(tmp)
            example['aug_examples'] = aug_examples
            return example

        new_dataset = dataset.add_column('aug_examples', [[dataset[i]] for i in range(len(dataset))])
        new_dataset = new_dataset.map(sample_ner, load_from_cache_file=False, num_proc=32)  # revise as needed

        examples = []
        for aug_examples in new_dataset['aug_examples']:
            examples.extend(aug_examples)

        new_dataset = Dataset.from_pandas(pd.DataFrame(examples, columns=columns)).shuffle(seed=self.seed,
                                                                                           load_from_cache_file=False)
        return new_dataset

    def generate_question(self, input_texts: Dataset, num_entries_per_input: int = 2,
                          batch_size: int = 16, log_every: int = 10000) -> Dataset:

        num_instructions = batch_size // num_entries_per_input

        sampler = BatchSampler(SequentialSampler(input_texts), batch_size=num_instructions, drop_last=False)
        dataset = []
        new_dataset = []
        log_count = 1
        columns = self.processor.dataset[self.processor.train_key].format['columns']

        for i, indices in enumerate(tqdm(sampler)):
            batch = [input_texts[i] for i in indices]
            to_add = self._generate_dataset_entries(batch,
                                                    num_samples=num_entries_per_input)

            new_dataset += postprocess_dataset(to_add)

            overall_size = len(dataset) + len(new_dataset)
            if self.processor and overall_size >= log_count * log_every:
                logging.info("Start using generated 1k data!")
                old_dataset = dataset
                res_dict = {}

                # combine the new dataset with old dataset
                dataset = old_dataset + new_dataset
                table = wandb.Table(data=pd.DataFrame(new_dataset[:100]))
                res_dict.update({'#Train': len(dataset), "examples": table})

                # re-init model and fine-tune from scratch
                self.processor.load_model()  # use the initial model

                logging.info("Train the model with full dataset.")
                self.processor.train(*self.processor.load_train_val(Dataset.from_pandas(pd.DataFrame(dataset,
                                                                                                     columns=columns)),
                                                                    seed=self.seed))  # use default params
                logging.info(f"Test results using {len(dataset)} training data: ")

                logging.info("Evaluate on validation dataset with new model.")
                val_metric = self.processor.validate()

                res_dict.update({"val": val_metric})

                logging.info(res_dict)
                wandb.log(res_dict)

                log_count += 1
                new_dataset = []

                logging.info("Save to disk...")
                Dataset.from_pandas(pd.DataFrame(dataset, columns=columns)).save_to_disk(self.output_dir)

        dataset += new_dataset
        dataset = Dataset.from_pandas(pd.DataFrame(dataset, columns=columns))
        return dataset

    def _generate_dataset_entries(self, batch: List[Dict], num_samples: int) -> List[Dict]:

        instructions = [self.instruction.replace(PLACEHOLDER_CONTEXT, example['context'])
                            .replace(PLACEHOLDER_ANSWER, example['answers']['text'][0])
                        for example in batch]

        model_outputs = self.model.generate_self_debiasing(
            input_texts=instructions,
            debiasing_texts=[],
            num_samples=num_samples,
            min_length=self.max_length,
            max_length=self.max_length,
            **self.generate_params
        )
        outputs = []
        for i, example in enumerate(batch):
            for j in range(num_samples):
                text = postprocess_question(example,
                                            output_text=model_outputs[i * num_samples + j],
                                            min_length=self.min_length)
                if text is not None:
                    example['question'] = text
                    example['id'] = example['id'] + '-' + str(j)
                    outputs.append(example)
        return outputs


def postprocess_question(example: Dict, output_text: str, min_length: int) -> Optional[str]:
    # a question should end with "?"
    if '?' in output_text:
        output_text = output_text.split('?')[0] + "?"
    else:
        return None

    # a question should not contain the answer
    if example['answers']['text'][0].lower() in output_text.lower():
        return None

    if len(output_text.strip().split(' ')) >= min_length:
        return output_text
    return None


def postprocess_answer(output_text: str, min_length: int) -> Optional[str]:
    if '"' in output_text:
        output_text = output_text.split('"')[0]
    elif '.' in output_text:
        output_text = output_text.split('.')[0]
    else:
        return None
    if len(output_text.strip().split(' ')) >= min_length:
        return output_text
    return None


def postprocess_dataset(dataset: List[Dict]):
    json_list = [json.dumps(i) for i in dataset]
    postprocessed_dataset = [json.loads(i) for i in list(dict.fromkeys(json_list))]
    return postprocessed_dataset
