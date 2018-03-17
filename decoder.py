# -*- coding: utf-8 -*-
import functools
import random

import torch as tc
import torch.nn.utils.rnn as rnn_utils
import torch.autograd as ag

import model_utils as mu

from torch import nn
from collections import namedtuple


DecoderForward = namedtuple(
    'DecoderForward',
    [
        'seq_softmaxes',
        'seq_chosen_ixs',
        'output_sents',
        'outputs',
        'hiddens',
        'is_taught'
    ],
)


class DecoderRNN(nn.Module):

    """Decoder RNN for the AutoEncoder.

    .. note::

        * Only supports a single layer
        * Hidden dim = enc_hidden_dim * enc_num_dir

        +-----------------------------------------------------------------+
        |                                                                 |
        | +--------------+    +--------------+     +--------------------+ |
        | |   Previous   |    |   Encoder    |     | Previous predicted | |
        | | hidden state |    | final hidden |     |   word embedding   | |
        | +------+-------+    +----+---------+     +----------+-----^---+ |
        |        |                 |                          |     |     |
        |        |                 |       +-------------+    |     |     |
        |        |                 +-------> Concatenate <----+     |     |
        |        |                         +-------+-----+          |     |
        |        |                                 |                |     |
        |        |         +--------------+        |                |     |
        |        +--------->  LSTM Step   <--------+                |     |
        |                  +------+-------+                         |     |
        |                         |                                 |     |
        |                         |                                 |     |
        |              +----------v-----------+                     |     |
        |              |       Linear:        |                     |     |
        |              | lstm hidden -> Vocab |                     |     |
        |              +----------+-----------+                     |     |
        |                         |                                 |     |
        |                         |                                 |     |
        |                  +------v------+                          |     |
        |                  |   Softmax   +--------------------------+     |
        |                  +-------------+                                |
        |                                                                 |
        +-----------------------------------------------------------------+
    """

    def __init__(
            self, embed_dim, hidden_dim, lang, max_len,
            embedding_matrix=None, embedding=None,
            verbose=False,
    ):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.lang = lang
        self.verbose = verbose
        self.max_len = max_len

        vocab_size = len(lang.w2ix)
        self.embedding = mu.build_embedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            embedding=embedding,
            embedding_matrix=embedding_matrix,
        )
        self.lstm = nn.LSTM(
            # We need to use the encoder hidden in all the time steps.
            #    So instead of using `h_2 = tanh(W_h*h_1, W_x*x)`
            #    We let's concatenate the encoder hidden with the input
            #    `h_2 = tanh(W_h*h_1, W_xp*[x, h_enc])`
            input_size=(hidden_dim+embed_dim),
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.vocab_softmax = nn.Sequential(
            nn.Linear(hidden_dim, vocab_size),
            nn.LogSoftmax(),
        )

    def forward(
            self, enc_hidden, true_sents=None, teacher_forcing=0.,
            is_training=False, ret_output_sents=False):
        """Perform forward prop through decoder for testing..

        input_embed: embedding of '<SOS>'
        :param enc_hidden: LstmHidden(
            h=(batch_sz, enc_hidden_dim),
            c=(batch_sz, enc_hidden_dim),
        )
        :type enc_hidden: Variable
        :param true_sents: Only valid during training ->
            Ex. ``[['<SOS>', 'This', 'is', 'me', '.','<EOS>', '<PADDING>'], ...]``
        :type true_sents: BatchSentIxs or NoneType
        :param teacher_forcing: Only valid during training
        :type teacher_forcing: float
        """
        # Assign variables used in the function
        input_embed = _get_sos_embeds(
            embedding=self.embedding,
            batch_size=enc_hidden.h.size()[0],
            lang=self.lang,
        )
        embedding = self.embedding
        max_len = self.max_len
        lang = self.lang
        lstm = self.lstm
        vocab_softmax = self.vocab_softmax
        # ------------------------------------------
        _check_fwd_prop_inputs(
            true_sents=true_sents,
            teacher_forcing=teacher_forcing,
            is_training=is_training,
        )
        # See :func:`_get_sos_embeds` for dim
        batch_sz = input_embed.size()[0]
        # The first word is not generated but is <SOS>
        gen_seq_len = max_len - 1
        hidden = (enc_hidden.h.unsqueeze(0), enc_hidden.c.unsqueeze(0))
        # Dim -> (batch_size, 1, hidden_sz)
        enc_h_3d = enc_hidden.h.unsqueeze(dim=1)
        # seq_len*[(batch, vocab_sz)]
        seq_softmaxes = [None] * max_len
        # seq_len*[(batch,)]
        seq_chosen_ixs = [
            [lang.w2ix['<SOS>']] * batch_sz
        ] + ([None] * gen_seq_len)
        # seq_len*[(batch,)] | Ex. [ [I, He], [am, is], [drunk, wasted]]
        output_sents = [['<SOS>'] * batch_sz] + ([None] * gen_seq_len)
        hiddens = [hidden] + ([None] * gen_seq_len)
        outputs = [None] * max_len
        is_taught = [False] + [None] * gen_seq_len

        for i in xrange(1, max_len):
            # dim -> (batch_sz, seq=1, embed_sz+hidden_sz)
            # args -> [list of tensors], dimension
            lstm_input = tc.cat((input_embed, enc_h_3d), dim=2)
            #log.info('lstm_input.size: ', lstm_input.size())
            # output->(batch, seq_len=1, hidden_size)
            output, hidden = lstm(lstm_input, hidden)
            #log.info('output.size: ', output.size(), 'hidden[0].size(): ', hidden[0].size())
            # dim -> (batch_size, vocab_size)
            yhat_i = vocab_softmax(output.squeeze())
            #log.info('yhat_i.size: ', yhat_i.size())
            # Chosen word ixs -> (batch_sz,)
            _, yhat_ixs_i = yhat_i.max(dim=1)
            #log.info('yhat_ixs_i ', yhat_ixs_i)
            if _coin_toss(p=teacher_forcing):
                #log.info('Teacher forced')
                is_taught[i] = True
                _yhat_ixs_i = [
                    true_sents.seqs.data[b][i]
                    for b in xrange(batch_sz)
                ]
                #log.info('_yhat_ixs_i: ', _yhat_ixs_i)
            else:
                #log.info('Not teacher forced')
                is_taught[i] = False
                _yhat_ixs_i = yhat_ixs_i
            # dim -> (batch_sz, 1, embed_dim)
            input_embed = _ixs2embed_for_rnn(
                ixs=_yhat_ixs_i,
                embedding=embedding,
            )
            # Save for returning
            hiddens[i] = hidden
            outputs[i] = output
            seq_softmaxes[i] = yhat_i
            seq_chosen_ixs[i] = yhat_ixs_i
            if ret_output_sents:
                output_sents[i] = [lang.ix2w[ix] for ix in yhat_ixs_i.data]

        return DecoderForward(
            seq_softmaxes=seq_softmaxes,
            seq_chosen_ixs=seq_chosen_ixs,
            output_sents=(
                fmt_dec_out_sents(sents=output_sents)
                if ret_output_sents
                else output_sents
            ),
            outputs=outputs,
            hiddens=hiddens,
            is_taught=is_taught,
        )


def _get_sos_embeds(embedding, batch_size, lang):
    # Dim (batch_size, seq_len=1, embed_dim)
    return _ixs2embed_for_rnn(
        ixs=[lang.w2ix['<SOS>']] * batch_size,
        embedding=embedding,
    )


def _ixs2embed_for_rnn(ixs, embedding):
    """Converts indices of a batch for next step of rnn.

    ixs -> (batch_sz,) or list of ixs of len batch_sz

    returns: (batch_sz, 1, embed_dim)
    """
    if isinstance(ixs, ag.Variable):
        # dim -> (batch_sz, 1)
        ix_var = ixs.unsqueeze(dim=1)
    else:
        ix_var = mu.cudable_variable(
            # dim -> (batch_sz, 1)
            tc.LongTensor(ixs).unsqueeze(dim=1),
            requires_grad=False,
        )
    return embedding(ix_var)


def _coin_toss(p):
    return random.random() < p


def _check_fwd_prop_inputs(true_sents, teacher_forcing, is_training):
    if is_training:
        assert true_sents is not None and teacher_forcing is not None
    else:
        assert true_sents is None and not teacher_forcing


def fmt_dec_out_sents(sents):
    # transpose sentences
    sents = map(list, zip(*sents))
    sents_fmt = []
    for s in sents:
        new_s = []
        for w in s:
            new_s.append(w)
            if w == '<EOS>':
                break
        sents_fmt.append(new_s)
    return sents_fmt
