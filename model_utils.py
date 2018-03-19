# -*- coding: utf-8 -*-
"""Contains model training utils and to convert sentences to indices.
"""
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


class BatchSentIxs(namedtuple('BatchSentIxs', ['seqs', 'lengths'])):

    def __len__(self):
        return len(self.seqs)

########################################################################
#                   Model training
########################################################################


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

def save_model(model, path):
    with open(path, 'wb') as outfile:
        pickle.dump(model, outfile, pickle.HIGHEST_PROTOCOL)


def load_model(path):
    with open(path, 'rb') as infile:
        return pickle.load(infile)


def build_embedding(vocab_size, embed_dim, embedding, embedding_matrix):
    if embedding is not None:
        return embedding
    embedding = nn.Embedding(vocab_size, embed_dim)
    if embedding_matrix is not None:
        embedding.weight.data.copy_(tc.from_numpy(embedding_matrix))
    return embedding


def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses over epochs for GD.')
    plt.show()


########################################################################
#                   Sentence representation
########################################################################

def seqs2var(seqs, lengths):
    """Given a list of list of indices and lengths create BatchSentIxs

    :param seqs: List of list of indices of the embedding matrix
    :type seqs: list(list(int)) or array(array(int))
    :param lengths: The actual length of the padded sentences.
    :type lengths: list(int) or array(int)

    :returns: BatchSentIxs sorted in descending order of the sent
        length as this is expected by the rnn.utils
        pack_padded_sequence.

    Sample Usage::

        print seqs2var(
            seqs=[
                [0, 3, 5, 7, 8, 1, 2, 2],
                [0, 3, 4, 4, 8, 1, 2, 2],
            ],
            lengths=[4,4],
        )

    Sample Output::

        BatchSentIxs(
            seqs=Variable containing:
                0     3     5     7     8     1     2     2
                0     3     4     4     8     1     2     2
                [torch.LongTensor of size 2x8]
            ,
            lengths=(4, 4),
        )
    """
    assert isinstance(seqs[0][0], int)
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

    :param sents: List of list of words (str). If BatchSentIxs
        we reutrn as is.
    """
    if isinstance(sents, BatchSentIxs):
        return sents
    if isinstance(sents[0], (str, unicode)):
        sents = [sents]
    # Make sure that we did not pass in the indexes of the w2v and
    #   we passed in the words
    assert isinstance(sents[0][0], (str, unicode))
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
    unk_ix = w2ix['<UNK>']
    for i, sent in enumerate(sents):
        if len(sent) > allowable_len:
            n_truncated += 1
            sent = sent[:allowable_len]
        unpadded_tokens = (
            [w2ix['<SOS>']] +
            [w2ix.get(w, unk_ix) for w in sent] +
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


def batch_ixs_to_sents(batch_sent_ixs, lang):
    """BatchSentIxs to sentences.

    Sample Usage::
        x = BatchSentIxs(
            seqs=Variable containing:
                0     4     5     7     9     1     2     2
                0     4     5     3     9     1     2     2
            [torch.LongTensor of size 2x8]
            ,
            lengths=(4, 4)
        )
        w2v_data.lang.ix2w = OrderedDict([(0, '<SOS>'),
             (1, '<EOS>'),
             (2, '<PADDING>'),
             (3, '<UNK>'),
             (4, 'i'),
             (5, 'love'),
             (6, 'ham'),
             (7, 'pizza'),
             (8, 'dragons'),
             (9, '.')])
        print batch_ixs_to_sents(d['dev'], lang=w2v_data.lang)

    Sample Output::

        [['<SOS>', 'i', 'love', 'pizza', '.', '<EOS>'],
         ['<SOS>', 'i', 'love', '<UNK>', '.', '<EOS>']]
    """
    return [
        [
            lang.ix2w[ix.data[0]]
            # Extra 2 chars for <SOS> and <EOS>
            for ix in ixs[:length + 2]
        ]
        for ixs, length in zip(
            batch_sent_ixs.seqs,
            batch_sent_ixs.lengths,
        )
    ]
