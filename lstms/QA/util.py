import json
import os
import nltk
import torch

from torchtext import data


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class SQuAD():
    def __init__(self, args):
        train_examples_path = os.path.dirname(args.train_file)+'/train.pt'
        dev_examples_path = os.path.dirname(args.dev_file)+'/dev.pt'
        test_examples_path = os.path.dirname(args.test_file)+'/test.pt'

        print("preprocessing data files...")
        if not os.path.exists('{}l'.format(args.train_file)):
            self.preprocess_file(args.train_file)
        if not os.path.exists('{}l'.format(args.dev_file)):
            self.preprocess_file(args.dev_file)
        if not os.path.exists('{}l'.format(args.test_file)):
            self.preprocess_file(args.test_file)

        self.RAW = data.RawField()
        # explicit declaration for torchtext compatibility
        self.RAW.is_target = False
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}

        list_fields = [('id', self.RAW), ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('q_word', self.WORD), ('q_char', self.CHAR)]

        if os.path.exists(train_examples_path):
            print("loading train splits...")
            train_examples = torch.load(train_examples_path)
            self.train = data.Dataset(examples=train_examples, fields=list_fields)
        else:
            print("building train splits...")
            self.train = data.TabularDataset.splits(
                path=os.path.dirname(args.train_file),
                train='{}l'.format(os.path.basename(args.train_file)),
                format='json',
                fields=dict_fields)[0]
            torch.save(self.train.examples, train_examples_path)

        if os.path.exists(dev_examples_path):
            print("loading dev splits...")
            dev_examples = torch.load(dev_examples_path)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
        else:
            print("building dev splits...")
            self.dev = data.TabularDataset.splits(
                path=os.path.dirname(args.dev_file),
                validation='{}l'.format(os.path.basename(args.dev_file)),
                format='json',
                fields=dict_fields)[0]
            torch.save(self.dev.examples, dev_examples_path)

        if os.path.exists(test_examples_path):
            print("loading test splits...")
            test_examples = torch.load(test_examples_path)
            self.test = data.Dataset(examples=test_examples, fields=list_fields)
        else:
            print("building test splits...")
            self.test = data.TabularDataset.splits(
                path=os.path.dirname(args.test_file),
                validation='{}l'.format(os.path.basename(args.test_file)),
                format='json',
                fields=dict_fields)[0]
            torch.save(self.test.examples, test_examples_path)

        #cut too long context in the training set for efficiency.
        if args.context_threshold > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= args.context_threshold]

        print("building vocab...")
        self.CHAR.build_vocab(self.train, self.dev)
        # self.WORD.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=args.word_dim))
        self.WORD.build_vocab(self.train, self.dev, max_size=50000)

        print("building iterators...")
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        self.train_iter = data.BucketIterator(
            self.train,
            batch_size=args.train_batch_size,
            device=device,
            repeat=True,
            shuffle=True,
            sort_key=lambda x: len(x.c_word)
        )

        self.dev_iter = data.BucketIterator(
            self.dev,
            batch_size=args.dev_batch_size,
            device=device,
            repeat=False,
            sort_key=lambda x: len(x.c_word)
        )

        self.test_iter = data.BucketIterator(
            self.test,
            batch_size=args.dev_batch_size,
            device=device,
            repeat=False,
            sort_key=lambda x: len(x.c_word)
        )

    def preprocess_file(self, path):
        dump = []
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']

            for article in data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    tokens = word_tokenize(context)
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(answer)

                            l = 0
                            s_found = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in abnormals:
                                        l += 1
                                    else:
                                        break
                                # exceptional cases
                                if t[0] == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\'' + t[1:]
                                elif t == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\''

                                l += len(t)
                                if l > s_idx and s_found == False:
                                    s_idx = i
                                    s_found = True
                                if l >= e_idx:
                                    e_idx = i
                                    break

                            dump.append(dict([('id', id),
                                              ('context', context),
                                              ('question', question),
                                              ('answer', answer),
                                              ('s_idx', s_idx),
                                              ('e_idx', e_idx)]))

        with open('{}l'.format(path), 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)
