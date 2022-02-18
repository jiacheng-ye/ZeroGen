from datasets import load_metric, load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import os
from .glue_processor import GLUEProcessor


class SST2Processor(GLUEProcessor):
    """Using the original sst-2 dataset (#standard train: 6290)"""
    def __init__(self, task_name, model_name, model_ckpt, output_dir, device, **train_args):
        super().__init__(task_name, model_name, model_ckpt, output_dir, device, **train_args)

    def load_model(self):
        self.num_labels = 2
        self.is_regression = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        load_name = self.model_ckpt if self.model_ckpt is not None else self.model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(load_name, num_labels=self.num_labels).to(self.device)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    def load_dataset(self):
        data_path = f'data/sst-2'
        if os.path.exists(data_path):
            self.dataset = load_from_disk(data_path)
        else:
            self.dataset = load_dataset('gpt3mix/sst2')
            self.dataset.save_to_disk(data_path)

        for name, subset in self.dataset.items():
            self.dataset[name] = subset.add_column('idx', list(range(len(subset))))

        self.sentence1_key, self.sentence2_key = 'text', None
        self.encoded_dataset = self._encode_dataset(self.dataset)
        self.metric = load_metric("glue", "sst2")
        self.main_metric_name = "eval_accuracy"

    def preprocess_function(self, examples):
        examples[self.sentence1_key] = [x.replace('-LRB-', '(').replace('-RRB-', ')')
                                        for x in examples[self.sentence1_key]]
        return self.tokenizer(examples[self.sentence1_key], truncation=True, max_length=512)




