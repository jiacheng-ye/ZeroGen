import torch.nn as nn
import math


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.1, layer_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        if layer_num == 1:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, bidirectional=True)

        else:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, dropout=dropout_rate,
                                  bidirectional=True)
        self.init_weights()

    def init_weights(self):
        for p in self.bilstm.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                p.data.zero_()
                # This is the range of indices for our forget gates for each LSTM cell
                p.data[self.hidden_size // 2: self.hidden_size] = 1

    def forward(self, x, lens):
        '''
        :param x: (batch, seq_len, input_size)
        :param lens: (batch, )
        :return: (batch, seq_len, hidden_size)
        '''
        # output, (ht, ct) = self.bilstm(x, )
        # sent_emb = ht[-2:].permute(1, 0, 2).reshape(len(lens), -1)
        # return output, sent_emb

        ordered_lens, index = lens.sort(descending=True)
        ordered_x = x[index]

        packed_x = nn.utils.rnn.pack_padded_sequence(ordered_x, ordered_lens.cpu(), batch_first=True)
        packed_output, (ht, ct) = self.bilstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        recover_index = index.argsort()
        recover_output = output[recover_index]

        sent_emb = ht[-2:].permute(1, 0, 2).reshape(len(lens), -1)
        sent_emb = sent_emb[recover_index]
        return recover_output, sent_emb


class RNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_size, hidden_size, dropout_rate, num_layers,
                 pretrained_embed=None, freeze=False):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pretrained_embed = pretrained_embed

        if pretrained_embed is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_embed, freeze)
        else:
            self.embed = nn.Embedding(vocab_size, embed_size)
            nn.init.uniform_(self.embed.weight.data,
                             a=-math.sqrt(3 / self.embed.weight.data.size(1)),
                             b=math.sqrt(3 / self.embed.weight.data.size(1)))

        self.rnn = BiLSTM(embed_size, hidden_size, dropout_rate, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_weights()

    def init_weights(self):
        if self.pretrained_embed is None:
            nn.init.uniform_(self.embed.weight.data,
                             a=-math.sqrt(3 / self.embed.weight.data.size(1)),
                             b=math.sqrt(3 / self.embed.weight.data.size(1)))

        nn.init.xavier_normal_(self.fc.weight.data)

    def forward(self, x, lens):
        embeddings = self.embed(x)
        output, sent_emb = self.rnn(embeddings, lens)
        out = self.fc(self.dropout(sent_emb))
        return out