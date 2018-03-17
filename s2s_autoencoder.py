# -*- coding: utf-8 -*-
"""Sequence to Sequence AutoEncoder model to learn sentence embeddings.

Author: Vishnu Sreenivasan | visp@yelp.com

Example usage::

    s2s_ae = Seq2SeqAutoEncoder(
        lang=lang,
        config=Seq2SeqConfig(
            max_len=24,
            embed_dim=4,
            hidden_dim=20,
            vocab_size=len(lang.w2ix),
            bidirectional=True,
            num_layers=2,
            lr=0.05,
        ),
    )
    s2sfit = s2s_ae.fit(
        training_data=[training_data],
        epochs=10,
        teacher_forcing=0.5,
        verbose=True,
        plot_losses=True,
    )
    print s2sfit.losses[-1])
    s2s_ae.forward(sents=training_data).dec_fwd.output_sents
    s2s_ae.encode_sentences(sents=training_data)
    s2s_ae.decode_sentences(
        s2s_ae.encode_sentences(sents=training_data),
    )
"""
import functools
import itertools
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import tqdm

import torch

import torch as tc
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.utils.rnn as rnn_utils

import encoder
import decoder
import model_utils as mu

from collections import namedtuple, OrderedDict
from itertools import chain


log = logging.getLogger(__name__)

DATASET_SPLITS = ['train', 'dev']
S2SForward = namedtuple('S2SForward', ['enc_fwd', 'dec_fwd'])
Seq2SeqConfig = namedtuple(
    'Seq2SeqConfig',
    (
        'max_len',
        'embed_dim',
        'hidden_dim',
        'vocab_size',
        'bidirectional',
        'num_layers',
        'lr',
    ),
)
EPOCHS_RESULT_TEMPL = """

---------------------------------------------------------------------

EPOCH RESULT: {epoch_num}
=================

{results}

---------------------------------------------------------------------

"""

#####################################################################
#                        Sequence to Sequence
#####################################################################
#                        Training Pseudocode
#
# ``` python
# embedding = mu.build_embedding()
# encoder = Encoder(embedding, **encoder_params)
# decoder = Decoder(embedding, **decoder_params)
#
# for [sent] in iter(data):
#     [sent] -> BatchSentIxs -> encoder -> EncoderForward
#     EncoderForward.h_last_layer -> decoder -> (softmaxes, sents)
#     loss = compute_loss(softmaxes, BatchSentIxs)
#     loss.backward()
#     encoder.step()
#     decoder.step()
# ```

class Seq2SeqAutoEncoder(nn.Module):

    """Seq2Seq autoencoder model to learn sentence embeddings

    See module docstring for example usage.

    """

    def __init__(
        self,
        lang,
        config,
        embedding=None,
        embedding_matrix=None,
        verbose=False,
        eval_fns=None,
    ):

        super(Seq2SeqAutoEncoder, self).__init__()

        assert config.hidden_dim % 4 == 0

        self.config = config
        self.lang = lang
        self.hidden_dim = config.hidden_dim//2
        self.sents2var = functools.partial(
            mu.sents2var,
            w2ix=lang.w2ix,
            max_sent_len=config.max_len,
        )

        # Model
        self.embedding = mu.build_embedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            embedding=embedding,
            embedding_matrix=embedding_matrix,
        )
        self.encoder = encoder.EncoderRNN(
            embed_dim=config.embed_dim,
            # We want to merge H and C of the LSTM hidden
            # as the final output.
            hidden_dim=self.hidden_dim,
            vocab_size=config.vocab_size,
            bidirectional=config.bidirectional,
            num_layers=config.num_layers,
            verbose=verbose,
            embedding=self.embedding,
        )
        self.decoder = decoder.DecoderRNN(
            embed_dim=config.embed_dim,
            # We want to merge H and C of the LSTM hidden
            # as the final output.
            hidden_dim=self.hidden_dim,
            lang=lang,
            max_len=config.max_len,
            verbose=verbose,
            embedding=self.embedding,
        )

        # Training params
        self.optimizer = optim.Adamax(params=self.parameters(), lr=config.lr)
        # self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
        #     optimizer=self.optimizer,
        #     factor=0.5,
        #     patience=1,
        #     verbose=True,
        #     threshold=0.1,
        #     threshold_mode='abs',
        #     cooldown=0,
        #     min_lr=1e-10,
        #     eps=1e-08,
        # )
        self.loss_fn = nn.NLLLoss()
        self.eval_fns = [compute_accuracy] if eval_fns is None else eval_fns

    def forward(
            self, sents=None, batch_sent_ixs=None, teacher_forcing=0.):
        """Forward propagation throughout the entire network

        Either ``sents`` or ``batch_sent_ixs`` should be
        passed but not both.

        :param batch_sents: List of tokenized sentences
        :type batch_sents: list(list(str)) or NoneType
        :param batch_sent_ixs: The tokenized, padded batch of
            sentences.
        :type batch_sent_ixs: BatchSentIxs or NoneType

        Sample ``batch_sents``::

            [
                ['<SOS>', 'i', 'am', 'a', 'black', 'cat', '.', '<EOS>'],
                ...
            ]
        """
        return s2s_forward(
            sents=sents,
            batch_sent_ixs=batch_sent_ixs,
            teacher_forcing=teacher_forcing,
            encoder_fwd_func=self.encoder.forward,
            decoder_fwd_func=self.decoder.forward,
            sents2var_func=self.sents2var,
        )

    def fit(self, *args, **kwargs):
        return s2s_fit(s2s_model=self, *args, **kwargs)

    def decode_sentences(self, enc_hiddens):
        """Given encoder hiddens, returns a sentence as a str.

        :param enc_hiddens: Dim: (batch_size, enc_hidden_dim*(enc_bidirectional+1))
        :type enc_hiddens: ag.Variable(tc.FloatTensor) or np.array

        :returns: A list of sentences of length enc_hiddens.size()[0].
        :rtype: list(str)
        """
        return s2s_decode_sentences(
            decoder_fwd_func=self.decoder.forward,
            enc_hiddens=enc_hiddens,
            hidden_dim=self.hidden_dim,
        )

    def encode_sentences(self, sents):
        """Gives the embedding for a list of sentences

        :param sents: A list of sentence tokens.
        :type sents: list(list(str))

        :returns: Sentence embeddings for all the sentences.
            Dim -> (len(sents), enc_hidden_dim*(enc_bidirectional+1))
        :rtype: np.array
        """
        return s2s_encode_sentences(
            sents=sents,
            encoder_fwd_func=self.encoder.forward,
            sents2var_func=self.sents2var,
        )


def s2s_forward(
    sents,
    batch_sent_ixs,
    teacher_forcing,
    encoder_fwd_func,
    decoder_fwd_func,
    sents2var_func,
):
    assert bool(sents) or bool(batch_sent_ixs)
    if not batch_sent_ixs:
        batch_sent_ixs = sents2var_func(sents=sents)
    # Forward prop
    enc_fwd = encoder_fwd_func(sents=batch_sent_ixs)
    dec_fwd = decoder_fwd_func(
        enc_hidden=enc_fwd.h_last_layer,
        true_sents=batch_sent_ixs,
        teacher_forcing=teacher_forcing,
        is_training=True,
        ret_output_sents=True,
    )
    return S2SForward(enc_fwd=enc_fwd, dec_fwd=dec_fwd)


def s2s_fit(
    s2s_model,
    dataloader,
    save_epoch_freq,
    model_save_path=None,
    epochs=1,
    teacher_forcing=0.5,
    tb_expt=None,
):
    """
    s2s_model: :class:`Seq2SeqAutoEncoder`
    dataloader: :class:`SentDataloader` yields train_sents, dev_sents
        train_sents, dev_sents: :class:`preprocessing.BatchSentIxs`
    save_epoch_freq: If 0 or None, we never save the model.
    save_path: If save_epoch_freq is truthy, this must be a valid path.
    """
    if save_epoch_freq:
        assert model_save_path is not None
    epoch_stats_list = []
    for epoch_num in tqdm.tqdm_notebook(range(epochs), desc='Epochs'):
        epoch_stats = run_epoch(
            s2s_model=s2s_model,
            dataloader=dataloader,
            teacher_forcing=teacher_forcing,
        )
        log_losses(
            epoch_stats=epoch_stats,
            tb_expt=tb_expt,
            epoch_num=epoch_num,
        )
        epoch_stats_list.append(epoch_stats)
        # TODO: Dont do any lr decay
        # s2s_model.lr_scheduler.step(e_loss)
        if save_epoch_freq and epoch_num % save_epoch_freq == 0:
            log.info('Saving model to `{}`...'.format(model_save_path))
            mu.save_model(model=s2s_model, path=model_save_path)
            log.info('Saved model to `{}`.'.format(model_save_path))
    return epoch_stats_list


def run_epoch(s2s_model, dataloader, teacher_forcing):
    """Train a single epoch of the model.

    :returns:

        {
            'split_metrics_map': {
                'train': {'loss': 3, 'evaluation/acc': 0.1, 'evaluation/s': 0.3},
                'dev': {'loss': 3, 'evaluation/acc': 2.1, 'evaluation/s': 1.3},
            },
            'grad_by_wts': {'mean': 2.1, 'std': 1.3},
        }
    """
    metrics = {
        s: {'loss': 0, 'evaluations': []}
        for s in DATASET_SPLITS
    }
    grad_by_wts = []
    for train_dev_batch in tqdm.tqdm_notebook(dataloader, desc='Batches'):
        for split, batch in train_dev_batch.iteritems():
            # Get data
            batch_sent_ixs = s2s_model.sents2var(sents=batch)
            # Zero the grads
            s2s_model.zero_grad()
            # Forward prop
            s2s_fwd = s2s_model.forward(
                batch_sent_ixs=batch_sent_ixs,
                teacher_forcing=(
                    teacher_forcing if split == 'train' else 0.
                ),
            )
            # Loss
            loss_dict = compute_loss(
                true_sents=batch_sent_ixs,
                seq_softmaxes=s2s_fwd.dec_fwd.seq_softmaxes,
                loss_fn=s2s_model.loss_fn,
                eval_fns=s2s_model.eval_fns,
            )
            loss = loss_dict['loss']
            metrics[split]['loss'] += loss.data[0]
            metrics[split]['evaluations'].append(loss_dict['evaluations'])
            if split == 'train':
                # Backward prop
                loss.backward()
                grad_by_wts.append(_compute_grad_by_wt(model=s2s_model))
                # Grad descent
                s2s_model.optimizer.step()
    return {
        'split_metrics_map': _agg_epoch_metrics(
            eval_fns=s2s_model.eval_fns,
            metrics=metrics,
        ),
        'grad_by_wts': _agg_epoch_grad_by_wts(grad_by_wts=grad_by_wts),
    }


def _agg_epoch_metrics(eval_fns, metrics):
    epoch_metrics = {}
    for split, split_metrics in metrics.iteritems():
        epoch_metrics[split] = {'loss': split_metrics['loss']}
        evals = np.mean(split_metrics['evaluations'], axis=0)
        for eval_fn, eval_val in zip(eval_fns, evals):
            epoch_metrics[split][
                'evaluation/{}'.format(eval_fn.__name__)
            ] = eval_val
    return epoch_metrics


def _agg_epoch_grad_by_wts(grad_by_wts):
    """
    grad_by_wts: [(mean_grad_by_wt, std_grad_by_wt)] # len: batch_sz
    """
    return {
        agg: np.mean(vals)
        for agg, vals in zip(['mean', 'std'], zip(*grad_by_wts))
    }


def log_losses(epoch_stats, tb_expt, epoch_num):
    log_data = {}
    for split, metrics in epoch_stats['split_metrics_map'].iteritems():
        for ident, metric_val in metrics.iteritems():
            log_data['{s}/{m}'.format(s=split, m=ident)] = metric_val
    for avg_or_std, val in epoch_stats['grad_by_wts'].iteritems():
        log_data['grad_by_wts/{a}'.format(a=avg_or_std)] = val
    tb_expt.add_scalar_dict(log_data)
    log.info(
        EPOCHS_RESULT_TEMPL.format(
            epoch_num=epoch_num,
            results='\n'.join(
                '{nm}: {val:,.4f}'.format(nm=nm, val=val)
                for nm, val in log_data.iteritems()
            ),
        ),
    )


def _compute_grad_by_wt(model):
    grad_rats = []
    for p in model.parameters():
        if p is None:
            continue
        p_data, p_grad = p.data.abs(), p.grad.data.abs()
        rat = (p_grad / p_data)
        rat[p_grad==0.] = 0.
        grad_rats.extend(rat.view(-1))
    return (np.mean(grad_rats), np.std(grad_rats))


def s2s_encode_sentences(sents, encoder_fwd_func, sents2var_func):
    last_layer = encoder_fwd_func(
        sents=sents2var_func(sents=sents),
    ).h_last_layer
    return last_layer#tc.cat((last_layer.h, last_layer.c), dim=1)


def s2s_decode_sentences(decoder_fwd_func, enc_hiddens, hidden_dim):
    # lasy_layer = mu.LstmHidden(
    #     # This is how it is trained, so this is okay.
    #     h=enc_hiddens[:,:hidden_dim],
    #     c=enc_hiddens[:,hidden_dim:],
    # )
    return decoder_fwd_func(
        enc_hidden=enc_hiddens,
        true_sents=None,
        teacher_forcing=None,
        is_training=False,
        ret_output_sents=True,
    ).output_sents


#####################################################################
#                        Loss
#####################################################################

def compute_loss(true_sents, seq_softmaxes, loss_fn, eval_fns):
    """ Cross entropy loss for the decoder.

    :seq_softmaxes dec_fwd.seq_softmaxes: list(Variable(batch_sz, vocab))
    :true_sents: BatchSentIxs
    """
    # batch_sz x seq_len x vocab_sz
    # Skip the first word as it is None. See decoder forward as the
    #   first word is assumed to be <SOS>.
    batch_y_preds = tc.cat(
        [s.unsqueeze(dim=1) for s in seq_softmaxes[1:]],
        dim=1,
    )
    loss = 0
    evaluations = np.zeros((len(true_sents.seqs), len(eval_fns)))
    for i, (sent_y_pred, sent_true_pred, sent_len) in enumerate(
        zip(
            batch_y_preds,
            true_sents.seqs,
            true_sents.lengths,
        )
    ):
        # Why upto -1 ???
        y_pred_relevant = sent_y_pred[:sent_len-1]
        # Drop the first <SOS> word
        y_true_relevant = sent_true_pred[1:sent_len]
        loss += loss_fn(y_pred_relevant, y_true_relevant)
        for j, each_eval_fn in enumerate(eval_fns):
            evaluations[i, j] = each_eval_fn(
                softmaxes=y_pred_relevant,
                true_ixs=y_true_relevant,
            )
    return {'loss': loss, 'evaluations': evaluations.mean(axis=0)}


def compute_accuracy(softmaxes, true_ixs):
    """

    batch_softmax: Variable (batch_sz, vocab_sz): (100, 400000)
    true_ixs: Variable (batch_sz,)
    """
    _, predicted = tc.max(softmaxes, dim=1)
    return (predicted == true_ixs).type(tc.DoubleTensor).mean()
