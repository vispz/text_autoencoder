# -*- coding: utf-8 -*-
import codecs
import logging
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch as tc
import torch.nn as nn
import tqdm

import model_utils as mu

from collections import OrderedDict
from collections import namedtuple
from itertools import izip
from sklearn.model_selection import train_test_split



log = logging.getLogger(__name__)

Lang = namedtuple('Lang', ('w2ix', 'ix2w'))
W2VData = namedtuple('W2VData', ('lang', 'embedding_matrix'))
SplitData = namedtuple('SplitData', ['train', 'dev', 'test'])

SPECIAL_TOKENS = ['<SOS>', '<EOS>', '<PADDING>', '<UNK>']

########################################################################
#                       Word Vectors
########################################################################

def load_w2v_data(
        path, num_lines=None, special_tokens=None, plot_embed=False):
    num_lines = num_lines or _count_lines(fl=path)
    special_tokens = special_tokens or SPECIAL_TOKENS
    log.debug('Special tokens used: `{st}`'.format(st=SPECIAL_TOKENS))
    n_special_tokens = len(special_tokens)
    embedding_matrix = []
    ix2w = OrderedDict(enumerate(special_tokens))
    w2ix = OrderedDict((tok, idx) for idx, tok in enumerate(special_tokens))

    with open(path) as f:
        for line_num, line in tqdm.tqdm(
            enumerate(f, n_special_tokens),
            total=num_lines,
            desc='Loading embeddings',
        ):
            values = line.split()
            word = values[0]
            embedding_matrix.append(map(float, values[1:]))
            ix2w[line_num] = word
            w2ix[word] = line_num
        embedding_matrix = _add_spl_toks_to_emb_mat(
            embedding_matrix=np.array(embedding_matrix, dtype=np.float32),
            n_special_tokens=n_special_tokens,
        )
    log.info(
        'Shape of embedding matrix `{0}`'.format(embedding_matrix.shape),
    )
    if plot_embed:
        plot_embed_matrix(mat=embedding_matrix)
    return W2VData(
        embedding_matrix=embedding_matrix,
        lang=Lang(ix2w=ix2w, w2ix=w2ix),
    )


def _count_lines(fl):
    with open(fl) as infile:
        return sum(1 for line in infile)


def _add_spl_toks_to_emb_mat(embedding_matrix, n_special_tokens):
    # Initialize the word vectors of unseen words to the
    # random normal with the mean and the std deviation
    # of each embedding column. A slightly similar approach
    # was mentioned in http://www.aclweb.org/anthology/D14-1181.
    # In this paper the author samples from U[-a, a] such
    # that the variance of each column remains the same.
    extra_tokens_w2v_init = np.random.multivariate_normal(
        mean=embedding_matrix.mean(axis=0).ravel(),
        cov=np.diag(embedding_matrix.std(axis=0).ravel()),
        size=n_special_tokens,
    )
    return np.concatenate(
        (extra_tokens_w2v_init, embedding_matrix),
        axis=0,
    )


def plot_embed_matrix(mat):
    plt.bar(range(mat.shape[1]), mat.mean(axis=0))
    plt.title('The mean of each dim')
    plt.xlabel('Dimensions')
    plt.ylabel('Mean')
    plt.show()


def build_w2ix(words):
    w2ix, ix2w = OrderedDict(), OrderedDict()
    words = SPECIAL_TOKENS + list(words)
    for w in words:
        if w in w2ix:
            continue
        new_ix = len(w2ix)
        w2ix[w] = new_ix
        ix2w[new_ix] = w
    return Lang(w2ix=w2ix, ix2w=ix2w)


def to_words(ixs, lang):
    return [lang.ix2w[ix] for ix in ixs]


########################################################################
#                       Review processing
########################################################################


def load_and_split_reviews(
        review_fl, lang, dev_frac=0.2, test_frac=0.2,
        n_reviews=None, max_len=None, min_len=1):
    train_dev_df, test_df = train_test_split(
        load_review_file_to_df(
            path=review_fl,
            spacy_nlp=get_spacy_nlp(),
            n_reviews=n_reviews,
            lang=lang,
            max_len=max_len,
            min_len=min_len,
        ),
        test_size=test_frac,
    )
    train_df, dev_df = train_test_split(train_dev_df, test_size=dev_frac)
    train_df, dev_df, test_df = map(_sort_df, (train_df, dev_df, test_df))
    return SplitData(train=train_df, dev=dev_df, test=test_df)


def load_review_file_to_df(
        path, lang, spacy_nlp=None, n_reviews=None, max_len=None,
        min_len=1):
    return _sort_df(
        df=pd.DataFrame(
            _iter_file_sent_tokens(
                path=path,
                lang=lang,
                nlp=spacy_nlp or get_spacy_nlp(),
                n_reviews=n_reviews,
                max_len=max_len,
                min_len=min_len,
            ),
        ),
    )


def _sort_df(df):
    df = df.sort_values(['review_id', 'sent_num']).reset_index(drop=True)
    return df[['review_id', 'sent_num', 'sent_len', 'ixs', 'tokens', 'text']]


def get_spacy_nlp():
    nlp = spacy.load('en')
    nlp.pipeline = [nlp.parser]
    return nlp


def _iter_file_sent_tokens(
        path, lang, nlp, n_reviews=None, max_len=None, min_len=1):
    """Iterate over the review file and yield sentence tokens.

    :param path: Path to the reviews file.
    :type path: str
    :param nlp: Spacy tokenizer with parsing.
    :type nlp: spacy.en.English
    :param n_reviews: Maximum number of reviews to iter over. If
        None we iter over all reviews.
    :type n_reviews: int or NoneType

    :returns: Iterator over a dict. See sample output.
    :rtype: iter(dict)

    Sample output::

        {'review_id': 'iamid', 'sent_num': 0, 'tokens': ['Every', 'villain', 'is', 'lemons']}
    """
    unk_ix = lang.w2ix['<UNK>']
    invalid_toks = {u' ', u'\n\n'}
    for review_dict, review_doc in izip(
        iter_review_dict(path=path, n_reviews=n_reviews),
        nlp.pipe(
            iter_review_text(path=path, n_reviews=n_reviews),
            batch_size=500,
            n_threads=4,
        ),
    ):
        for sent_num, sent in enumerate(review_doc.sents):
            _tokens = [
                unicode(tok.lower_)
                for tok in sent
                if tok.lower_ not in invalid_toks
            ]
            if max_len:
                _tokens = _tokens[:max_len-2]
                tokens = ['<SOS>'] + _tokens + ['<EOS>']
                tokens = tokens + ['<PADDING>'] * (max_len - len(tokens))
            else:
                tokens = ['<SOS>'] + _tokens + ['<EOS>']
            sent_len = min(max_len, len(sent))
            if sent_len >= min_len:
                yield {
                    'sent_num': sent_num,
                    'text': u' '.join(map(unicode, sent)),
                    'tokens': tokens,
                    'review_id': review_dict['review_id'],
                    # Plus 2 because we are including the <SOS> and
                    # <EOS>.
                    'sent_len': min(max_len, len(sent) + 2),
                    'ixs': [lang.w2ix.get(t, unk_ix) for t in tokens],
                }


def iter_review_dict(path, n_reviews=None):
    with codecs.open(path, encoding='utf-8') as infile:
        for i, line in tqdm.tqdm_notebook(
            enumerate(infile),
            total=n_reviews,
            desc='Iter over reviews.',
        ):
            if n_reviews and n_reviews == i:
                return
            yield json.loads(line)


def iter_review_text(path, n_reviews=None):
    for review_dict in iter_review_dict(path=path, n_reviews=n_reviews):
        yield review_dict['text']
