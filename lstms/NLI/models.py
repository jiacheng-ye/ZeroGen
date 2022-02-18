import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TwostreamBilstm(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_size, hidden_size, dropout_rate=0.1, layer_num=1,
                 pretrained_embed=None, freeze=False):
        super(TwostreamBilstm, self).__init__()
        self.pretrained_embed = pretrained_embed
        if pretrained_embed is not None:
            self.embeddings = nn.Embedding.from_pretrained(pretrained_embed, freeze)
        else:
            self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.directions = 2
        self.concat = 4
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=layer_num, bidirectional=True,
                            dropout=dropout_rate, batch_first=True)
        self.hidden2label = nn.Linear(hidden_size, num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.lin1 = nn.Linear(self.hidden_size * self.directions * self.concat, self.hidden_size)
        self.lin2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin3 = nn.Linear(self.hidden_size, num_labels)

        for lin in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.out = nn.Sequential(
            self.lin1,
            self.relu,
            self.dropout,
            self.lin2,
            self.relu,
            self.dropout,
            self.lin3
        )

    def forward(self, x1, x1_lens, x2, x2_lens):
        '''
        :param x1: (batch, seq1_len)
        :param x1_lens: (batch,)
        :param x2: (batch, seq2_len)
        :param x2_lens: (batch,)
        :return: (batch, num_class)
        '''
        # Input encoding
        lengths_a = x1_lens.cpu()
        lengths_b = x2_lens.cpu()
        batch_size, seq_len = x1.size()
        x_a = self.embeddings(x1)
        x_b = self.embeddings(x2)
        packed_input_a = pack_padded_sequence(x_a, lengths_a, enforce_sorted=False, batch_first=True)
        packed_input_b = pack_padded_sequence(x_b, lengths_b, enforce_sorted=False, batch_first=True)
        lstm_out_a, (ht_a, ct_a) = self.lstm(packed_input_a)
        lstm_out_b, (ht_b, ct_b) = self.lstm(packed_input_b)

        output_padded_a, output_lengths_a = pad_packed_sequence(lstm_out_a, batch_first=True)
        output_padded_b, output_lengths_b = pad_packed_sequence(lstm_out_b, batch_first=True)
        out_a = output_padded_a.sum(dim=1).div(output_lengths_a.float().unsqueeze(dim=1).cuda())
        out_b = output_padded_b.sum(dim=1).div(output_lengths_b.float().unsqueeze(dim=1).cuda())

        premise = out_a.contiguous().view(batch_size, -1)
        hypothesis = out_b.contiguous().view(batch_size, -1)

        combined = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis), 1)
        logits = self.out(combined)
        return logits
