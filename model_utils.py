# -*- coding: utf-8 -*-
import cPickle as pickle
import logging

import matplotlib.pyplot as plt
import torch as tc
import torch.autograd as ag
import torch.nn as nn

from collections import namedtuple

from torch.utils import data as tc_data_utils

log = logging.getLogger(__name__)
IS_CUDA = tc.cuda.is_available()
print 'is_cuda:', IS_CUDA

LstmHidden = namedtuple('LstmHidden', ('h', 'c'))
BatchSentIxs = namedtuple('BatchSentIxs', ['seqs', 'lengths'])


class SentDataset(tc_data_utils.Dataset):

    def __init__(self, train_sent_df, dev_sent_df):
        cols = ['sent_len', 'ixs']
        self.dfs_map = {
            'train': train_sent_df[cols],
            'dev': dev_sent_df[cols],
        }
        self.szs = {nm: len(df) for nm, df in self.dfs_map.iteritems()}

    def __len__(self):
        return self.szs['train']

    def __getitem__(self, ix):
        assert ix < self.szs['train']
        return {
            # We want to cycle over the dev and the test set
            ident: df.ix[ix % self.szs[ident]].to_dict()
            for ident, df in self.dfs_map.iteritems()
        }


def cudable_variable(*args, **kwargs):
    var = ag.Variable(*args, **kwargs)
    if IS_CUDA:
        return var.cuda()
    else:
        return var


def dataloader_collate_fn(batch):
    batch_sz = len(batch)
    cltd_batch = {
        split: {it: [None] * batch_sz for it in ('ixs', 'sent_len')}
        for split in batch[0]
    }
    for i, row in enumerate(batch):
        for split, ixs_n_lens in row.iteritems():
            for nm, val in ixs_n_lens.iteritems():
                cltd_batch[split][nm][i] = val
    return {
        split: seqs2var(seqs=v['ixs'], lengths=v['sent_len'])
        for split, v in cltd_batch.iteritems()
    }


def to_dataloader(
        train_sent_df, dev_sent_df, batch_size=4, num_workers=4,
        drop_last=False, **kwargs):
    return tc_data_utils.DataLoader(
        SentDataset(
            train_sent_df=train_sent_df,
            dev_sent_df=dev_sent_df,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataloader_collate_fn,
        **kwargs
    )


def build_embedding(vocab_size, embed_dim, embedding, embedding_matrix):
    if embedding is not None:
        return embedding
    embedding = nn.Embedding(vocab_size, embed_dim)
    if embedding_matrix is not None:
        embedding.weight.data.copy_(tc.from_numpy(embedding_matrix))
    return embedding


def seqs2var(seqs, lengths):
    # The data is passed in. We assume that
    #    BatchSentIxs(seqs=np.array, lengths=np.array)
    # The padded sequence should in sorted in reversed
    #    sentence length
    lengths, seqs = zip(*sorted(zip(lengths, seqs), reverse=True))
    return BatchSentIxs(
        seqs=cudable_variable(tc.LongTensor(seqs), requires_grad=False),
        # unpadded sentence lengths
        lengths=lengths,
    )


def sents2var(sents, w2ix=None, max_sent_len=None):
    """Function to convert a batch of sentences into padded input variable.
    """
    if isinstance(sents, BatchSentIxs):
        return sents
    if isinstance(sents[0], str):
        sents = [sents]
    sent_lens, tokens_list = _sents2ixs(
        sents=sents,
        w2ix=w2ix,
        max_sent_len=max_sent_len
    )
    return BatchSentIxs(
        seqs=cudable_variable(
            tc.LongTensor(tokens_list),
            requires_grad=False,
        ),
        # unpadded sentence lengths
        lengths=sent_lens,
    )


def _sents2ixs(sents, w2ix, max_sent_len):
    """Helper function used by sents2var.
    """
    n_truncated = 0
    # We need two tokens for sentence start and sentence end
    allowable_len = max_sent_len - 2
    # [(len(sent_tok_ixs), sent_tok_ixs), ..]
    # Ex: [['hello', 'world'], ...] ->
    #    [['<SOS>', 'hello', 'world', '<EOS>', '<pad>', '<pad>'], ...] ->
    #    [(4, [0, 25, 42, 1, 2, 2]), ...]
    tokens_sentlen_list = [None] * len(sents)
    sent_lens = []
    for i, sent in enumerate(sents):
        if len(sent) > allowable_len:
            n_truncated += 1
            sent = sent[:allowable_len]
        unpadded_tokens = (
            [w2ix['<SOS>']] +
            [w2ix.get(w, '<UNK>') for w in sent] +
            [w2ix['<EOS>']]
        )
        sent_len = len(unpadded_tokens)
        padded_tokens = (
            unpadded_tokens +
            ([w2ix['<PADDING>']] * (max_sent_len-sent_len))
        )
        tokens_sentlen_list[i] = (sent_len, padded_tokens)
    sent_lens, tokens_list = zip(*sorted(tokens_sentlen_list, reverse=True))
    if n_truncated:
        log.info('Had to truncate `{l}` sentences'.format(l=n_truncated))
    return sent_lens, tokens_list


def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses over epochs for GD.')
    plt.show()


def save_model(model, path):
    with open(path, 'wb') as outfile:
        pickle.dump(model, outfile, pickle.HIGHEST_PROTOCOL)


def load_model(path):
    with open(path, 'rb') as infile:
        return pickle.load(infile)
