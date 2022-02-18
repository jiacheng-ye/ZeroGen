#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import torch
from utils import set_seed, read_jsonl
import logging
import os
from main import task2processor
from datasets import load_from_disk
from cls_generator import convert_to_hf_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default='sst-2',
                        help="The output directory for storing the trained model and evaluation results.")
    parser.add_argument("--output_dir", type=str, default='tmp',
                        help="The output directory for storing the trained model and evaluation results.")

    # Model and training parameters
    parser.add_argument("--dataset", type=str, default=None,
                        help="Local dataset to use.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use limited samples to use for training.")
    parser.add_argument("--no_train", action='store_true',
                        help="Only evaluate.")
    parser.add_argument("--small_model_name", type=str, default='distilbert-base-uncased',
                        help="The pretrained Transformer language model to use.")
    parser.add_argument("--small_model_ckpt", type=str, default=None,
                        help="The saved model to load.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs to train the small model.")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Size of batch to train the small model.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate to train the small model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed used to initialize all random number generators.")

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args.seed)
    print(f"Parameters: {args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = task2processor(args.task_name)(task_name=args.task_name,
                                               model_name=args.small_model_name,
                                               model_ckpt=args.small_model_ckpt,
                                               output_dir=args.output_dir,
                                               device=device,
                                               num_epochs=args.num_epochs,
                                               train_batch_size=args.train_batch_size,
                                               learning_rate=args.learning_rate
                                               )

    if args.dataset:  # pseudo/synthetic dataset
        if args.task_name == 'squad' or args.task_name == 'adversarial_qa':
            dataset = load_from_disk(args.dataset)
            if args.limit:
                dataset = dataset.select(range(args.limit))
        else:
            dataset = read_jsonl(args.dataset)
            if args.limit:
                dataset = dataset[:args.limit]
            dataset = convert_to_hf_dataset(dataset, processor.sentence1_key, processor.sentence2_key)
    else:  # standard dataset
        dataset = processor.dataset['train']
        if args.limit:
            dataset = dataset.select(range(args.limit))

    processor.train(*processor.load_train_val(dataset, seed=args.seed), train=not args.no_train)
    val_metric = processor.validate()
    logging.info("Metric on standard dataset: " + str(val_metric))
    val_metric = processor.validate(dataset)
    logging.info("Metric on synthetic dataset: " + str(val_metric))