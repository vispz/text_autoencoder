# -*- coding: utf-8 -*-
import functools

import torch as tc
import torch.nn.utils.rnn as rnn_utils

import model_utils as mu

from torch import nn
from collections import namedtuple


EncoderForward = namedtuple(
    'EncoderForward',
    [
        'output',
        'lengths',
        'hidden',
        'h_last_layer',
    ],
)


class EncoderRNN(nn.Module):

    """Encoder RNN, converts sentence to emedding.

         +----------------+
         | Input sentence |
         +------+---------+
                |
                |
          +-----v------+
          |   Word     |
          | embedding  |
          +-----+------+
                |
                |
        +-------v---------+
        |      LSTM       |
        |Possibly stacked |
        +-------+---------+
                |
                |
        +-------v---------+
        |  Encoder output |
        +-----------------+
    """

    def __init__(
            self, embed_dim, hidden_dim, vocab_size,
            bidirectional=True, num_layers=1,
            embedding_matrix=None, embedding=None, verbose=False,
    ):
        super(EncoderRNN, self).__init__()

        self.hidden_dim = hidden_dim // (2 if bidirectional else 1)
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.verbose = verbose

        self.embedding = mu.build_embedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            embedding=embedding,
            embedding_matrix=embedding_matrix,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            # We want to pass in (batch, seq, feature)
            batch_first=True,
        )

    def forward(self, sents):
        """Forward prop.

        :param sents:
        :type sents: BatchSentIxs
        """
        batch_size = len(sents.lengths)

        # Get embedding & init hidden
        # Dim (batch_size, sent_len, embed_dim)
        embeds = self.embedding(sents.seqs)
        packed_embeds = rnn_utils.pack_padded_sequence(
            input=embeds,
            lengths=sents.lengths,
            batch_first=True,
        )
        hidden = self.init_hidden(batch_size=batch_size)

        # Apply LSTM
        output, hidden = self.lstm(packed_embeds, hidden)
        output, lengths = rnn_utils.pad_packed_sequence(
            sequence=output,
            batch_first=True,
        )
        hidden_last = pull_last_hidden(
            hidden=hidden,
            num_layers=self.num_layers,
            num_dir=(1 + self.bidirectional),
        )
        return EncoderForward(
            output=output,
            lengths=lengths,
            hidden=hidden,
            h_last_layer=mu.LstmHidden(
                # This is the sentence embedding :D
                h=hidden_last[0],
                c=hidden_last[1],
            ),
        )

    def init_hidden(self, batch_size):
        dims = (self.num_layers*(self.bidirectional+1),
                batch_size, self.hidden_dim)
        return (
            # Cell h_0 (num_layers * num_directions, batch, hidden_size)
            mu.CudableVariable(tc.zeros(*dims), requires_grad=True),
            # Hidden c_0 (num_layers * num_directions, batch, hidden_size)
            mu.CudableVariable(tc.zeros(*dims), requires_grad=True),
        )


def pull_last_hidden(hidden, num_layers, num_dir):
    """Get the last layers hidden state output.

    This handles padding :)

    Dim -> (batch_size, num_dir*hidden_dim)
    """
    pull_func = functools.partial(
        _pull_last_hidden_helper,
        num_layers=num_layers,
        num_dir=num_dir,
    )
    if isinstance(hidden, tuple):
        # LSTM
        return map(pull_func, hidden)
    else:
        # GRU or RNN
        return pull_func(h=hidden)


def _pull_last_hidden_helper(h, num_layers, num_dir):
    _, batch_size, hidden_dim = h.size()
    return h[(num_layers-1)*num_dir:].transpose(
        dim1=0,
        dim2=1,
    ).resize(batch_size, num_dir*hidden_dim)
