"""Data utilities."""
import os
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bert_serving.client import BertClient

import torch
from torch.autograd import Variable

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow_hub as hub
import sentencepiece as spm

from cuda import CUDA


class Vectorizer(object):
    def __init__(self):
        self.encodings = {}

    def fit(self, documents):
        return

    def transform(self, documents):
        encoded_documents = []
        for document in documents:
            if document in self.encodings:
                encoded_documents.append(self.encodings[document])
            else:
                encoding = self.encode(document)
                self.encodings[document] = encoding
                encoded_documents.append(encoding)
        return np.vstack(encoded_documents)


class BertVectorizer(Vectorizer):
    def __init__(self):
        super(BertVectorizer, self).__init__()
        self.bc = BertClient()

    def encode(self, document):
        return np.squeeze(self.bc.encode([document if document else " "]))


class TFHubVectorizer(Vectorizer):
    def __init__(self):
        super(TFHubVectorizer, self).__init__()
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def encode(self, document):
        return np.squeeze(self.sess.run(self.embed([document])))


class UniversalEncoderLargeVectorizer(TFHubVectorizer):
    def __init__(self):
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
        super(UniversalEncoderLargeVectorizer, self).__init__()


class UniversalEncoderVectorizer(TFHubVectorizer):
    def __init__(self):
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        super(UniversalEncoderVectorizer, self).__init__()


class UniversalEncoderLiteVectorizer(TFHubVectorizer):
    def __init__(self):
        self.input = tf.sparse_placeholder(tf.int64, shape=[None, None])
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
        self.encoding = self.embed(
            inputs=dict(
                values=self.input.values,
                indices=self.input.indices,
                dense_shape=self.input.dense_shape))
        super(UniversalEncoderLiteVectorizer, self).__init__()
        spm_path = self.sess.run(self.embed(signature="spm_path"))
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)

    def process_to_IDs_in_sparse_format(self, sentences):
        ids = [self.sp.EncodeAsIds(x) for x in sentences]
        max_len = max(len(x) for x in ids)
        dense_shape=(len(ids), max_len)
        values=[item for sublist in ids for item in sublist]
        indices=[[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
        return (values, indices, dense_shape)

    def encode(self, document):
        values, indices, dense_shape = self.process_to_IDs_in_sparse_format([document])
        return np.squeeze(self.sess.run(self.encoding,
            feed_dict={ self.input.values: values,
                        self.input.indices: indices,
                        self.input.dense_shape: dense_shape}))


class CorpusSearcher(object):
    def __init__(self, query_corpus, key_corpus, value_corpus, vectorizer, make_binary=True):
        self.query_corpus = query_corpus
        self.key_corpus = key_corpus
        self.value_corpus = value_corpus
        
        # rows = docs, cols = features
        self.vectorizer = vectorizer
        self.vectorizer.fit(key_corpus)
        self.key_corpus_matrix = self.vectorizer.transform(self.key_corpus)
        if make_binary:
            self.key_corpus_matrix = (self.key_corpus_matrix != 0).astype(int) # make binary

        
    def most_similar(self, key_idx, n=10):
        # print(self.key_corpus)
        # print("key_corpus_matrix: {}".format(self.key_corpus_matrix))
        # print(type(self.key_corpus_matrix))
        # print(self.key_corpus_matrix.shape)

        query = self.query_corpus[key_idx]
        query_vec = self.vectorizer.transform([query])
        # print(query)
        # print("query_vec: {}".format(query_vec))
        # print(type(query_vec))
        # print(query_vec.shape)

        scores = np.dot(self.key_corpus_matrix, query_vec.T)
        # print("scores: {}".format(scores))
        # print(type(scores))
        try:
            scores = np.squeeze(scores.toarray())
        except:
            scores = np.squeeze(scores)
        # print("post squeeze scores: {}".format(scores))
        # print(type(scores))
        # print(scores.shape)
        scores_indices = zip(scores, range(len(scores)))
        selected = sorted(scores_indices, reverse=True)[:n]

        # use the retrieved i to pick examples from the VALUE corpus
        selected = [
            (self.key_corpus[i], self.value_corpus[i], i, score) 
            for (score, i) in selected
        ]

        return selected


def build_vocab_maps(vocab_file):
    assert os.path.exists(vocab_file), "The vocab file %s does not exist" % vocab_file
    unk = '<unk>'
    pad = '<pad>'
    sos = '<s>'
    eos = '</s>'

    lines = [x.strip() for x in open(vocab_file)]

    assert lines[0] == unk and lines[1] == pad and lines[2] == sos and lines[3] == eos, \
        "The first words in %s are not %s, %s, %s, %s" % (vocab_file, unk, pad, sos, eos)

    tok_to_id = {}
    id_to_tok = {}
    for i, vi in enumerate(lines):
        tok_to_id[vi] = i
        id_to_tok[i] = vi

    # Extra vocab item for empty attribute lines
    empty_tok_idx =  len(id_to_tok)
    tok_to_id['<empty>'] = empty_tok_idx
    id_to_tok[empty_tok_idx] = '<empty>'

    return tok_to_id, id_to_tok


def extract_attributes(line, attribute_vocab):
    content = []
    attribute = []
    for tok in line:
        if tok in attribute_vocab:
            attribute.append(tok)
        else:
            content.append(tok)
    return line, content, attribute


def get_vectorizer(vectorizer_type, vocabulary):
    if vectorizer_type == "count":
        return CountVectorizer(vocabulary=vocabulary)
    elif vectorizer_type == "tfidf":
        return TfidfVectorizer(vocabulary=vocabulary)
    elif vectorizer_type == "bert":
        return BertVectorizer()
    elif vectorizer_type == "universal_encoder_large":
        return UniversalEncoderLargeVectorizer()
    elif vectorizer_type == "universal_encoder":
        return UniversalEncoderVectorizer()
    elif vectorizer_type == "universal_encoder_lite":
        return UniversalEncoderLiteVectorizer()
    else:
        raise Exception("Unknown vectorizer type: %s" % vectorizer_type)


def read_nmt_data(src, config, tgt, attribute_vocab, train_src=None, train_tgt=None):
    attribute_vocab = set([x.strip() for x in open(attribute_vocab)])

    src_lines = [l.strip().split() for l in open(src, 'r')]
    src_lines, src_content, src_attribute = list(zip(
        *[extract_attributes(line, attribute_vocab) for line in src_lines]
    ))
    src_tok2id, src_id2tok = build_vocab_maps(config['data']['src_vocab'])
    # train time: just pick attributes that are close to the current (using word distance)
    # we never need to do the TFIDF thing with the source because 
    # test time is strictly in the src => tgt direction
    src_dist_measurer = CorpusSearcher(
        query_corpus=[' '.join(x) for x in src_attribute],
        key_corpus=[' '.join(x) for x in src_attribute],
        value_corpus=[' '.join(x) for x in src_attribute],
        vectorizer=get_vectorizer(config['data']['src_vectorizer'], src_tok2id),
        make_binary=True if config['data']['src_vectorizer'] in ('count', 'tfidf') else False
    )
    src = {
        'data': src_lines, 'content': src_content, 'attribute': src_attribute,
        'tok2id': src_tok2id, 'id2tok': src_id2tok, 'dist_measurer': src_dist_measurer
    }

    tgt_lines = [l.strip().split() for l in open(tgt, 'r')] if tgt else None
    tgt_lines, tgt_content, tgt_attribute = list(zip(
        *[extract_attributes(line, attribute_vocab) for line in tgt_lines]
    ))
    tgt_tok2id, tgt_id2tok = build_vocab_maps(config['data']['tgt_vocab'])
    # train time: just pick attributes that are close to the current (using word distance)
    if train_src is None or train_tgt is None:
        tgt_dist_measurer = CorpusSearcher(
            query_corpus=[' '.join(x) for x in tgt_attribute],
            key_corpus=[' '.join(x) for x in tgt_attribute],
            value_corpus=[' '.join(x) for x in tgt_attribute],
            vectorizer=get_vectorizer(config['data']['tgt_train_vectorizer'], tgt_tok2id),
            make_binary=True if config['data']['src_vectorizer'] in ('count', 'tfidf') else False
        )
    # at test time, use test src-side content to scan through train tgt-side content
    #     (using tfidf) and retrieve corresponding attributes
    else:
        tgt_dist_measurer = CorpusSearcher(
            query_corpus=[' '.join(x) for x in src_content],
            key_corpus=[' '.join(x) for x in train_tgt['content']],
            value_corpus=[' '.join(x) for x in train_tgt['attribute']],
            vectorizer=get_vectorizer(config['data']['tgt_test_vectorizer'], tgt_tok2id),
            make_binary=False
        )
    tgt = {
        'data': tgt_lines, 'content': tgt_content, 'attribute': tgt_attribute,
        'tok2id': tgt_tok2id, 'id2tok': tgt_id2tok, 'dist_measurer': tgt_dist_measurer
    }

    return src, tgt

def sample_replace(lines, dist_measurer, sample_rate, corpus_idx):
    """
    replace sample_rate * batch_size lines with nearby examples (according to dist_measurer)
    not exactly the same as the paper (words shared instead of jaccaurd during train) but same idea
    """
    out = [None for _ in range(len(lines))]
    for i, line in enumerate(lines):
        if random.random() < sample_rate:
            sims = dist_measurer.most_similar(corpus_idx + i)[1:]  # top match is the current line
            try:
                line = next( (
                    tgt_attr.split() for tgt_cntnt, tgt_attr, _, _ in sims
                    if tgt_attr != ' '.join(line) # and tgt_attr != ''   # TODO -- exclude blanks?
                ) )
            # all the matches are blanks
            except StopIteration:
                line = []
            line = ['<s>'] + line + ['</s>']

        # corner case: special tok for empty sequences (just start/end tok)
        if len(line) == 2:
            line.insert(1, '<empty>')
        out[i] = line

    return out


def get_minibatch(lines, tok2id, index, batch_size, max_len, sort=False, idx=None,
        dist_measurer=None, sample_rate=0.0):
    """Prepare minibatch."""
    # FORCE NO SORTING because we care about the order of outputs
    #   to compare across systems
    lines = [
        ['<s>'] + line[:max_len] + ['</s>']
        for line in lines[index:index + batch_size]
    ]

    if dist_measurer is not None:
        lines = sample_replace(lines, dist_measurer, sample_rate, index)

    lens = [len(line) - 1 for line in lines]
    max_len = max(lens)

    unk_id = tok2id['<unk>']
    input_lines = [
        [tok2id.get(w, unk_id) for w in line[:-1]] +
        [tok2id['<pad>']] * (max_len - len(line) + 1)
        for line in lines
    ]

    output_lines = [
        [tok2id.get(w, unk_id) for w in line[1:]] +
        [tok2id['<pad>']] * (max_len - len(line) + 1)
        for line in lines
    ]

    mask = [
        ([1] * l) + ([0] * (max_len - l))
        for l in lens
    ]

    if sort:
        # sort sequence by descending length
        idx = [x[0] for x in sorted(enumerate(lens), key=lambda x: -x[1])]

    if idx is not None:
        lens = [lens[j] for j in idx]
        input_lines = [input_lines[j] for j in idx]
        output_lines = [output_lines[j] for j in idx]
        mask = [mask[j] for j in idx]

    input_lines = Variable(torch.LongTensor(input_lines))
    output_lines = Variable(torch.LongTensor(output_lines))
    mask = Variable(torch.FloatTensor(mask))

    if CUDA:
        input_lines = input_lines.cuda()
        output_lines = output_lines.cuda()
        mask = mask.cuda()

    return input_lines, output_lines, lens, mask, idx


def minibatch(src, tgt, idx, batch_size, max_len, model_type, is_test=False):
    if not is_test:
        use_src = random.random() < 0.5
        in_dataset = src if use_src else tgt
        out_dataset = in_dataset
        attribute_id = 0 if use_src else 1
    else:
        in_dataset = src
        out_dataset = tgt
        attribute_id = 1

    if model_type == 'delete':
        inputs = get_minibatch(
            in_dataset['content'], in_dataset['tok2id'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            out_dataset['data'], out_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

        # true length could be less than batch_size at edge of data
        batch_len = len(outputs[0])
        attribute_ids = [attribute_id for _ in range(batch_len)]
        attribute_ids = Variable(torch.LongTensor(attribute_ids))
        if CUDA:
            attribute_ids = attribute_ids.cuda()

        attributes = (attribute_ids, None, None, None, None)

    elif model_type == 'delete_retrieve':
        inputs =  get_minibatch(
            in_dataset['content'], in_dataset['tok2id'], idx, batch_size, max_len, sort=True)
        attributes =  get_minibatch(
            out_dataset['attribute'], out_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1],
            dist_measurer=out_dataset['dist_measurer'], sample_rate=0.25)
        outputs = get_minibatch(
            out_dataset['data'], out_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

    elif model_type == 'seq2seq':
        # ignore the in/out dataset stuff
        inputs = get_minibatch(
            src['data'], src['tok2id'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            tgt['data'], tgt['tok2id'], idx, batch_size, max_len, idx=inputs[-1])
        attributes = (None, None, None, None, None)

    else:
        raise Exception('Unsupported model_type: %s' % model_type)

    return inputs, attributes, outputs


def unsort(arr, idx):
    """unsort a list given idx: a list of each element's 'origin' index pre-sorting
    """
    unsorted_arr = arr[:]
    for i, origin in enumerate(idx):
        unsorted_arr[origin] = arr[i]
    return unsorted_arr
