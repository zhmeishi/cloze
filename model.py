import tensorflow as tf
import pandas as pd
import numpy as np
import nltk

import argparse
import datetime
import inflect
import logging
import pathlib
import pickle
import sys
import os
import re


# constants
BOS = '<bos>'
EOS = '<eos>'
PAD = '<pad>'
UNK = '<unk>'


def get_exp_path():
    '''Return new experiment path.'''

    return 'log/exp-{0}'.format(
        datetime.datetime.now().strftime('%m-%d-%H:%M:%S'))


def get_logger(path):
    '''Get logger for experiment.'''

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                  '%(levelname)s - %(message)s')

    # stdout log
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # file log
    handler = logging.FileHandler(path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def nltk_download():
    '''Download required resources for NLTK.'''

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')


def save_obj(objname, obj):
    '''Save python object.

    Args:
        objname: filename of pickle file
        obj: object to save
    '''

    pathlib.Path('tmp').mkdir(exist_ok=True)
    with open('tmp/%s.pickle' % objname, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(objname):
    '''Load python object from pickle file.

    Args:
        objname: name of the object

    Returns:
        Loaded object, None if no such object.
    '''

    if not os.path.exists('tmp/%s.pickle' % objname):
        return None

    obj = None
    with open('tmp/%s.pickle' % objname, 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_corpus(path, params, fmt):
    '''Load corpus.

    Args:
        path: corpus path
        params: experiment parameters
        fmt: 'train'/'val'

    Returns:
        fmt == 'train':
            A dictionary with key 'id', 'title', 'story'
        fmt == 'val':
            A dictionary with key 'id', 'context', 'endings', 'answer'
    '''

    assert fmt in ['train', 'val'], ('Unexpected value %s for `fmt`' % fmt)

    logger = logging.getLogger('__main__')
    logger.info('Loading %s corpus from %s' % (fmt, path))

    csv = pd.read_csv(path)
    if fmt == 'train':
        # training corpus
        train_corpus = load_obj('train_corpus')
        if train_corpus is not None:
            logger.info('Use saved train_corpus')
            return train_corpus

        storyid = list(csv['storyid'])
        title = list(csv['storytitle'])
        story = [list(csv['sentence%d' % i]) for i in range(1, 6)]

        # transpose and tokenize story
        story = list(zip(*story))
        story = [[nltk.word_tokenize(sen) for sen in sample]
                 for sample in story]

        train_corpus = {'id': storyid, 'title': title, 'story': story}
        save_obj('train_corpus', train_corpus)
        return train_corpus
    else:
        # validation corpus
        val_corpus = load_obj('val_corpus')
        if val_corpus is not None:
            logger.info('Use saved val_corpus')
            return val_corpus

        storyid = list(csv['InputStoryid'])
        context = [list(csv['InputSentence%d' % i]) for i in range(1, 5)]
        endings = [list(csv['RandomFifthSentenceQuiz%d' % i]) for i in [1, 2]]
        answer = list(csv['AnswerRightEnding'])

        # transpose and tokenize context
        context = list(zip(*context))
        context = [[nltk.word_tokenize(sen) for sen in sample]
                   for sample in context]

        # transpose and tokenize endings
        endings = list(zip(*endings))
        endings = [[nltk.word_tokenize(sen) for sen in sample]
                   for sample in endings]

        # answer id to zero-based
        answer = [(answer_id-1) for answer_id in answer]

        val_corpus = {'id': storyid, 'context': context, 'endings': endings, 'answer': answer}
        save_obj('val_corpus', val_corpus)
        return val_corpus


def preprocess_corpus(corpus, params, fmt):
    '''Preprocess the corpus.

    Args:
        corpus: corpus to preprocess
        params: experiment parameters
        fmt: 'train'/'val'

    Returns:
        Preprocessed corpus
    '''

    assert fmt in ['train', 'val'], ('Unexpected value %s for `fmt`' % fmt)

    logger = logging.getLogger('__main__')
    logger.info('Preprocessing %s corpus' % fmt)

    # load saved object
    objname = fmt + '_corpus_preprocessed'
    preprocessed = load_obj(objname)
    if preprocessed is not None:
        logger.info('Use saved %s' % objname)
        return preprocessed

    # helper function to transform each item of d[k] by f
    def transform(d, k, f):
        d[k] = [[f(sen) for sen in sample] for sample in d[k][:10]]

    # replace name of person to `Person`
    def replace_person(sentence):
        tree = nltk.ne_chunk(nltk.pos_tag(sentence))

        # traverse tree
        def visit(node):
            tokens = []
            for child in node:
                if type(child) is nltk.tree.Tree:
                    if child.label() == 'PERSON':
                        tokens.append('Person')
                    else:
                        tokens.extend(visit(child))
                else:
                    tokens.append(child[0])
            return tokens
        return visit(tree)

    # whether a string is int
    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    p = inflect.engine()

    # normalize a word (lowercasing + number to word + mapping)
    def normalize_word(word):
        word = word.lower()
        if is_int(word): word = p.number_to_words(word)
        if word == 'n\'t': word = 'not'
        elif word == '\'s': word = 's'
        return word

    # normalize a sentence (only contains a-z)
    def normalize_sentence(sen):
        sen = ' '.join(sen)
        sen = re.sub('[^a-z]', ' ', sen).split()

        # padding
        sen = [BOS] + sen
        sen = sen[:params.max_sentence_length]
        sen.append(EOS)
        sen.extend([PAD] * (params.max_sentence_length + 1 - len(sen)))
        return sen

    if fmt == 'train':
        # train corpus
        transform(corpus, 'story', replace_person)
        transform(corpus, 'story', lambda l: list(map(normalize_word, l)))
        transform(corpus, 'story', normalize_sentence)
    else:
        # val corpus
        transform(corpus, 'context', replace_person)
        transform(corpus, 'context', lambda l: list(map(normalize_word, l)))
        transform(corpus, 'context', normalize_sentence)
        transform(corpus, 'endings', replace_person)
        transform(corpus, 'endings', lambda l: list(map(normalize_word, l)))
        transform(corpus, 'endings', normalize_sentence)

    save_obj(objname, corpus)
    return corpus


def build_dictionary(corpus, params):
    '''Build dictionary from corpus.

    Args:
        corpus: corpus on which dictionary is built
        params: experiment parameters

    Returns:
        A dictionary mapping token to index
    '''

    logger = logging.getLogger(__name__)
    logger.info('Building dictionary from training corpus')

    dico = load_obj('dico')
    if dico is not None:
        logger.info('Use saved dictionary of size %d' % len(dico))
        return dico

    dico, token_cnt = {}, {}
    dico[BOS], dico[EOS], dico[PAD], dico[UNK] = 0, 1, 2, 3
    dico_size = len(dico)

    # count tokens
    for story in corpus['story']:
        for sentence in story:
            for token in sentence:
                # skip BOS/EOS/PAD
                if token in dico:
                    continue
                cnt = token_cnt.get(token, 0)
                token_cnt[token] = cnt + 1
    
    for token in sorted(token_cnt.keys(),
        key=lambda k: token_cnt[k], reverse=True):
        dico[token] = dico_size
        dico_size += 1
        if dico_size == params.vocab_size:
            break
    
    logger.info('Final size of dictionary is %d' % len(dico))
    save_obj('dico', dico)
    return dico


def transform_corpus(corpus, dico, params, fmt):
    '''Transform a corpus using a dictionary.
    
    Args:
        corpus: corpus to be transformed
        dico: a mapping from token to index
        params: experiment parameters
        fmt: 'train'/'val'

    Returns:
        A transformed corpus
    '''

    assert fmt in ['train', 'val'], ('Unexpected value %s for `fmt`' % fmt)

    logger = logging.getLogger(__name__)
    logger.info('Transforming %s corpus' % fmt)

    objname = fmt + '_corpus_transformed'
    transformed_corpus = load_obj(objname)
    if transformed_corpus is not None:
        logger.info('Use saved %s' % objname)
        return transformed_corpus

    # helper function to transform each item of d[k] by f
    def transform(d, k, f):
        d[k] = [[f(sen) for sen in sample] for sample in d[k][:10]]

    def word2id(sen):
        return list(map(lambda token: dico.get(token, dico[UNK]), sen))

    if fmt == 'train':
        # train corpus
        transform(corpus, 'story', word2id)
        corpus['story'] = np.array(corpus['story'], dtype=np.int32)
        logger.info('Train corpus story shape ' + str(corpus['story'].shape))
    else:
        # validation corpus
        transform(corpus, 'context', word2id)
        corpus['context'] = np.array(corpus['context'], dtype=np.int32)
        logger.info('Validation corpus context shape ' + str(corpus['context'].shape))
        transform(corpus, 'endings', word2id)
        corpus['endings'] = np.array(corpus['endings'], dtype=np.int32)
        logger.info('Validation corpus endings shape ' + str(corpus['endings'].shape))

    logger.info('Finished transforming corpus')
    save_obj(objname, corpus)
    return corpus


def main():
    '''Main function.'''

    # download NLTK resources
    nltk_download()

    # parser for command line arguments
    parser = argparse.ArgumentParser(description='NLU project 2')
    # network architecture
    parser.add_argument('--emb_dim', type=int, default=100,
                        help='Embedding dimension, default 100')
    parser.add_argument('--state_dim', type=int, default=512,
                        help='LSTM cell hidden state dimension (for c and h), default 512')
    parser.add_argument('--hidden_proj_dim', type=int, default=None,
                        help='Project hidden output before softmax, default None')
    # input data preprocessing
    parser.add_argument('--train_corpus', type=str, default='data/train.csv',
                        help='Path to training corpus')
    parser.add_argument('--val_corpus', type=str, default='data/val.csv',
                        help='Path to evaluation corpus')
    parser.add_argument('--max_sentence_length', type=int, default=40,
                        help='Maximum sentence length to keep, default 40')
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='Vocabulary size, default 20000')
    # experiment path
    parser.add_argument('--exp_path', type=str, default=None,
                        help='Experiment path')
    params = parser.parse_args()

    # parameter validation
    assert os.path.exists(params.train_corpus)
    assert os.path.exists(params.val_corpus)

    # experiment path
    if params.exp_path is None:
        # wait until time string is available for exp path
        while True:
            params.exp_path = get_exp_path()
            if not pathlib.Path(params.exp_path).exists():
                break
    pathlib.Path(params.exp_path).mkdir(parents=True, exist_ok=False)

    # logger
    logger = get_logger(params.exp_path + '/experiment.log')
    logger.info('Start of experiment')
    logger.info('============ Initialized logger ============')
    logger.info('\n\t' + '\n\t'.join('%s: %s' % (k, str(v))
        for k, v in sorted(dict(vars(params)).items())))

    # load training and validation corpora
    train_corpus = load_corpus(params.train_corpus, params, 'train')
    val_corpus = load_corpus(params.val_corpus, params, 'val')

    # preprocess corpora
    train_corpus = preprocess_corpus(train_corpus, params, 'train')
    val_corpus = preprocess_corpus(val_corpus, params, 'val')

    # build dictionary and transforming corpors
    dico = build_dictionary(train_corpus, params)
    train_corpus = transform_corpus(train_corpus, dico, params, 'train')
    val_corpus = transform_corpus(val_corpus, dico, params, 'val')


if __name__ == '__main__':
    main()
