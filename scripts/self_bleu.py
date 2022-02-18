import argparse
import json
import os
import random
from functools import partial
from multiprocessing.pool import Pool

import spacy
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--n_sample", type=int, default=1000,
                        help="how many sentences to sample to calculate bleu")
    parser.add_argument("--logto", type=str)
    parser.add_argument("--gen_column", type=str, required=True)

    return parser.parse_args()


def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)


def get_all_sentences(args):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

    if os.path.isfile(args.file):
        all_sentences = []
        with open(args.file, "r") as f_in:
            for line in f_in:
                obj = json.loads(line.strip())
                all_sentences.append(obj[args.gen_column])
    else:
        all_sentences = datasets.load_from_disk(args.file)[args.gen_column]

    all_sentences = random.sample(all_sentences, args.n_sample)
    all_sentences = [[tok.text for tok in nlp(s)] for s in all_sentences]
    return all_sentences


def main():
    args = parse_args()
    random.seed(0)
    all_sentences = get_all_sentences(args)
    smoothing_function = SmoothingFunction().method1

    pool = Pool(processes=os.cpu_count())
    bleu_scores = []
    for n_gram in range(1, 6):

        if n_gram == 1:
            weights = (1.0, 0, 0, 0)
        elif n_gram == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n_gram == 3:
            weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
        elif n_gram == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        elif n_gram == 5:
            weights = (0.2, 0.2, 0.2, 0.2, 0.2)
        else:
            raise ValueError
        bleu_scores.append(
            list(tqdm(
                pool.imap_unordered(
                    partial(bleu_i, weights, all_sentences, smoothing_function),
                    list(range(len(all_sentences)))),
                total=args.n_sample,
                smoothing=0.0,
                desc=f"bleu-{n_gram}")))
        print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / args.n_sample}")

    for n_gram in range(5):
        print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / args.n_sample}")

    if args.logto:
        with open(args.logto, 'a') as fout:
            print(f"{os.path.basename(args.file)}", end='\t', file=fout)
            for n_gram in range(5):
                print(f"{sum(bleu_scores[n_gram]) / args.n_sample}", end='\t', file=fout)
            print(file=fout)


if __name__ == '__main__':
    main()