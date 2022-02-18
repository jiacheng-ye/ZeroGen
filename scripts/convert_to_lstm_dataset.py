#!/usr/bin/python3
# -*- coding: utf-8 -*-

import datasets
import json
import sys
import os


def main_tc():
    in_path = f'{input_dir}/{task_name}-dataset.jsonl'
    out_path = f"{output_dir}/train.jsonl"

    with open(in_path, 'r', encoding='utf8') as fi:
        with open(out_path, 'w', encoding='utf8') as fo:
            for line in fi:
                line = json.loads(line)
                out = {'text': line['X'], 'label': line['Y']}
                fo.write(f'{json.dumps(out)}\n')


def main_nli():
    in_path = f'{input_dir}/{task_name}-dataset.jsonl'
    out_path = f"{output_dir}/train.jsonl"

    with open(in_path, 'r', encoding='utf8') as fi:
        with open(out_path, 'w', encoding='utf8') as fo:
            for line in fi:
                line = json.loads(line)
                out = {'premise': line['C'], 'hypothesis': line['X'], 'label': line['Y']}
                fo.write(f'{json.dumps(out)}\n')


def main_qa():
    print(f"loading from {input_dir}")
    dataset = datasets.load_from_disk(input_dir)

    total = len(dataset)

    print(f"random {limit} from {total}")
    assert limit <= total

    res_dict = {'data': [], 'version': '1.1'}
    data = {}
    for i in range(limit):
        data.setdefault(dataset[i]['title'], {})
        data[dataset[i]['title']].setdefault(dataset[i]['context'], [])

        answers = dataset[i]['answers']
        reshaped_answers = [{'answer_start': answer_start, 'text': text}
                            for answer_start, text in zip(answers['answer_start'], answers['text'])]
        qas = {'answers': reshaped_answers,
               'id': dataset[i]['id'],
               'question': dataset[i]['question']}
        data[dataset[i]['title']][dataset[i]['context']].append(qas)

    res_dict['data'] = [{'title': title,
                         'paragraphs': [{'context': context, 'qas': qas_list} for context, qas_list in
                                        paragraphs.items()]
                         }
                        for title, paragraphs in data.items()]

    total = len(res_dict['data'])
    train = {'data': res_dict['data'][:int(total * 0.9)], 'version': '1.1'}
    dev = {'data': res_dict['data'][int(total * 0.9):], 'version': '1.1'}

    train_path = f"{output_dir}/train-{limit}.json"
    print(f"save to {train_path}")
    with open(train_path, "w") as f:
        json.dump(train, f)

    dev_path = f"{output_dir}/dev-{limit}.json"
    print(f"save to {dev_path}")
    with open(dev_path, "w") as f:
        json.dump(dev, f)


if __name__ == '__main__':
    task_name = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    limit = int(sys.argv[4])  # unused for nli and tc

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if task_name in ['sst-2', 'imdb']:
        main_tc()
    elif task_name in ['qnli', 'rte']:
        main_nli()
    elif task_name in ['squad', 'adversarial_qa']:
        main_qa()
    else:
        raise RuntimeError()
