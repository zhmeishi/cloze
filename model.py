import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
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
CONTEXT_LENGTH = 4
SEED = 19980814


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
        fmt: 'train'/'val'/'test'/'submit'

    Returns:
        fmt == 'train':
            A dictionary with key 'id', 'title', 'story'
        fmt == 'val'/'test':
            A dictionary with key 'id', 'context', 'endings', 'answer'
        fmt == 'submit':
            A dictionary with key 'context', 'endings'
    '''

    assert fmt in ['train', 'val', 'test', 'submit'], (
        'Unexpected value %s for `fmt`' % fmt)

    logger = logging.getLogger('__main__')
    logger.info('Loading %s corpus from %s' % (fmt, path))

    if fmt == 'submit':
        csv = pd.read_csv(path, header=None, encoding='latin2')
    else:
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
    elif fmt in ['val', 'test']:
        # validation/test corpus
        val_corpus = load_obj(fmt + '_corpus')
        if val_corpus is not None:
            logger.info('Use saved %s_corpus' % fmt)
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

        val_corpus = {'id': storyid, 'context': context,
                      'endings': endings, 'answer': answer}
        save_obj(fmt + '_corpus', val_corpus)
        return val_corpus
    else:
        # submission corpus
        context = [list(csv[i]) for i in range(CONTEXT_LENGTH)]
        endings = [list(csv[i])
                   for i in range(CONTEXT_LENGTH, CONTEXT_LENGTH + 2)]

        # transpose and tokenize context
        context = list(zip(*context))
        context = [[nltk.word_tokenize(sen) for sen in sample]
                   for sample in context]

        # transpose and tokenize endings
        endings = list(zip(*endings))
        endings = [[nltk.word_tokenize(sen) for sen in sample]
                   for sample in endings]

        corpus = {'context': context, 'endings': endings}
        save_obj('submit_corpus', corpus)
        return corpus


def preprocess_corpus(corpus, params, fmt):
    '''Preprocess the corpus.

    Args:
        corpus: corpus to preprocess
        params: experiment parameters
        fmt: 'train'/'val'/'test'/'submit'

    Returns:
        Preprocessed corpus
    '''

    assert fmt in ['train', 'val', 'test', 'submit'], (
        'Unexpected value %s for `fmt`' % fmt)

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
        d[k] = [[f(sen) for sen in sample] for sample in d[k]]

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
        sen = sen[:(params.max_sentence_length-1)]
        sen.append(EOS)
        sen.extend([PAD] * (params.max_sentence_length + 1 - len(sen)))
        return sen

    if fmt == 'train':
        # train corpus
        transform(corpus, 'story', replace_person)
        transform(corpus, 'story', lambda l: list(map(normalize_word, l)))
        transform(corpus, 'story', normalize_sentence)
    else:
        # val/test/submit corpus
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

    logger = logging.getLogger('__main__')
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
                # skip BOS/EOS/PAD/UNK
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
        fmt: 'train'/'val'/'test'/'submit'

    Returns:
        A transformed corpus
    '''

    assert fmt in ['train', 'val', 'test', 'submit'], (
        'Unexpected value %s for `fmt`' % fmt)

    logger = logging.getLogger('__main__')
    logger.info('Transforming %s corpus' % fmt)

    objname = fmt + '_corpus_transformed'
    transformed_corpus = load_obj(objname)
    if transformed_corpus is not None:
        logger.info('Use saved %s' % objname)
        return transformed_corpus

    # helper function to transform each item of d[k] by f
    def transform(d, k, f):
        d[k] = [[f(sen) for sen in sample] for sample in d[k]]

    def word2id(sen):
        return list(map(lambda token: dico.get(token, dico[UNK]), sen))

    if fmt == 'train':
        # train corpus
        transform(corpus, 'story', word2id)
        corpus['story'] = np.array(corpus['story'], dtype=np.int32)
        logger.info('Train corpus story shape ' + str(corpus['story'].shape))
    else:
        # validation/test/submission corpus
        transform(corpus, 'context', word2id)
        corpus['context'] = np.array(corpus['context'], dtype=np.int32)
        logger.info('%s corpus context shape %s' %
                    (fmt, str(corpus['context'].shape)))
        transform(corpus, 'endings', word2id)
        corpus['endings'] = np.array(corpus['endings'], dtype=np.int32)
        logger.info('%s corpus endings shape %s' %
                    (fmt, str(corpus['endings'].shape)))

    logger.info('Finished transforming corpus')
    save_obj(objname, corpus)
    return corpus


def load_pretrained_embeddings(path, params):
    '''Load pretrained word embeddings.

    Args:
        path: path to pretrained word embeddings
        params: experiment parameters

    Returns:
        A numpy array with shape [dico_size, emb_dim]
    '''

    logger = logging.getLogger('__main__')
    logger.info('Loading pretrained embedding from %s' % params.pretrained)

    # read embedding
    logger.info('Reading file')
    embedding = np.empty(
        shape=[params.dico_size, params.emb_dim], dtype=np.float)
    found_tokens = set()
    with open(params.pretrained, 'r') as f:
        for i, line in enumerate(f):
            # early break
            if (params.max_pretrained_vocab_size is not None and
                i > params.max_pretrained_vocab_size):
                logger.info('Reach maximum pretrained vocab size %d' %
                            params.max_pretrained_vocab_size)
                break

            line = line.strip().split()
            if i == 0: # first line
                assert len(line) == 2, 'Invalid format at first line'
                _, dim = map(int, line)
                assert dim == params.emb_dim, 'Config to load embedding of ' \
                    'dimension %d but see %d' % (params.emb_dim, dim)
            else: # embedding line
                token = line[0]
                token_id = params.dico.get(token, -1)
                if token_id < 0: # not in dico
                    continue
                try:
                    embedding[token_id] = np.array(
                        list(map(float, line[1:])), dtype=np.float)
                    found_tokens.add(token)
                except ValueError:
                    continue

    # check unfound tokens
    logger.info('Checking unfound tokens')
    unfound_cnt = 0
    for token in params.dico.keys() - found_tokens:
        unfound_cnt += 1
        logger.info('Cannot load pretrained embedding for token %s' % token)
        embedding[params.dico[token]] = np.random.uniform(
            low=-0.25, high=0.25, size=params.emb_dim)

    logger.info('%d tokens not found in %s' % (unfound_cnt, path))
    logger.info('Finish loading pretrained embedding')
    return embedding


class SentenceEncoder:
    '''Sentence encoder using LSTM.'''

    def __init__(self, params):
        '''Constructor for SentenceEncoder.'''

        self.params = params
        state_dim = params.state_dim

        with tf.variable_scope('SentenceEncoder'):
            # embeddings
            if params.pretrained is not None:
                pretrained = load_pretrained_embeddings(
                    params.pretrained, params)
                pretrained = tf.convert_to_tensor(pretrained, tf.float32)
                self.embeddingW = tf.get_variable(
                    'embeddingW', None, tf.float32, pretrained, trainable=False)
            else:
                vocab_size, emb_dim = params.vocab_size, params.emb_dim
                self.embeddingW = tf.get_variable(
                    'embeddingW', [vocab_size, emb_dim], tf.float32,
                    tf.contrib.layers.xavier_initializer())

            # RNN cell
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(state_dim,
                initializer=tf.contrib.layers.xavier_initializer(),
                name='word_cell')

    def __call__(self, inputs):
        '''Encode input sentence.

        Args:
            inputs: `Tensor` of shape [sample_count, sentence_count, num_steps]

        Returns:
            Sentence embedding of shape [sample_count, sentence_count, state_dim]
        '''

        state_dim = self.params.state_dim
        num_steps = self.params.max_sentence_length

        # inputs has shape [sample_count, sentence_count, num_steps]
        sentence_count = tf.shape(inputs)[1]
        inputs = tf.reshape(inputs, [-1, num_steps])
        batch_size = tf.shape(inputs)[0]

        # [num_steps, batch_size, emb_dim]
        embedding = tf.nn.embedding_lookup(self.embeddingW, inputs)
        embedding = tf.transpose(embedding, [1, 0, 2])

        # sentence length
        eos_pos = tf.where(tf.equal(inputs,
            tf.constant(self.params.dico[EOS])))[:, 1]

        initial_state = self.rnn_cell.zero_state(batch_size, tf.float32)
        _, final_state = tf.nn.dynamic_rnn(self.rnn_cell, embedding,
            sequence_length=eos_pos, initial_state=initial_state,
            time_major=True)

        final_state = final_state.h
        final_state = tf.reshape(final_state, [-1, sentence_count, state_dim])
        return final_state


class StoryRNN:
    '''Story RNN.'''

    def __init__(self, params):
        '''Constructor for Story RNN.'''

        self.params = params
        state_dim = params.state_dim

        with tf.variable_scope('StoryRNN'):
            # sentence encoder
            self.encoder = SentenceEncoder(params)

            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(state_dim,
                initializer=tf.contrib.layers.xavier_initializer(),
                name='sentence_cell')

    def __call__(self, inputs, initial_state=None):
        '''Process a batch of stories.

        Args:
            inputs: `Tensor` of shape [batch_size, sentence_count, num_steps]
            initial_state: initial hidden states, default None

        Returns:
            LSTMStateTuple, c and h are of shape [batch_size, state_dim]
        '''

        sentence_embedding = self.encoder(inputs)
        batch_size = tf.shape(sentence_embedding)[0]

        # [sentence_count, batch_size, state_dim]
        sentence_embedding = tf.transpose(sentence_embedding, [1, 0, 2])

        if initial_state is None:
            initial_state = self.rnn_cell.zero_state(batch_size, tf.float32)
        _, final_state = tf.nn.dynamic_rnn(self.rnn_cell, sentence_embedding,
            initial_state=initial_state, time_major=True)

        return final_state

    def step(self, state, inputs):
        '''Make single step in RNN.

        Args:
            state: current hidden state, [batch_size, state_dim]
            inputs: next sentence, [batch_size, num_steps]

        Returns:
            New state with shape [batch_size, state_dim]
        '''

        # [batch_size, state_dim]
        state_dim = self.params.state_dim
        inputs = tf.expand_dims(inputs, 1)
        inputs = self.encoder(inputs)
        inputs = tf.reshape(inputs, [-1, state_dim])

        _, new_state = self.rnn_cell(inputs, state)
        return new_state


class ClozeClassifier:
    '''Classifier for story cloze test.'''

    def __init__(self, params):
        '''Constructor for the classifier.'''

        self.params = params
        num_steps = params.max_sentence_length
        state_dim = params.state_dim

        with tf.variable_scope('ClozeClassifier'):
            # train/eval mode
            self.training = tf.placeholder(tf.bool, [])

            # context sentences
            self.context = tf.placeholder(
                tf.int32, [None, None, num_steps])
            batch_size = tf.shape(self.context)[0]

            # possible endings and whether ending is true/false
            self.endings = tf.placeholder(tf.int32, [None, None, num_steps])
            self.input_y = tf.placeholder(tf.int32, [None, None])
            endings_per_context = tf.shape(self.endings)[1]

            # expand context cell/hidden state
            # shape [batch_size, endings_per_context, state_dim]
            self.story_encoder = StoryRNN(params)
            hidden_c, hidden_h = self.story_encoder(self.context)
            hidden_c = tf.tile(tf.reshape(
                hidden_c, [batch_size, 1, -1]), [1, endings_per_context, 1])
            hidden_c = tf.reshape(hidden_c, [-1, state_dim])
            hidden_h = tf.tile(tf.reshape(
                hidden_h, [batch_size, 1, -1]), [1, endings_per_context, 1])
            hidden_h = tf.reshape(hidden_h, [-1, state_dim])
            hidden = tf.contrib.rnn.LSTMStateTuple(hidden_c, hidden_h)

            # final encoded story
            # shape [batch_size * endings_per_context, num_steps]
            endings = tf.reshape(self.endings, [-1, num_steps])
            hidden = self.story_encoder.step(hidden, endings)
            hidden = hidden.h

            if params.clf_hidden is not None:
                hidden_dims = [int(x) for x in params.clf_hidden.split('-')]
                for dim in hidden_dims:
                    hidden = tf.layers.dropout(
                        hidden, self.params.dropout, training=self.training)
                    hidden = tf.layers.dense(hidden, dim, tf.nn.relu)

            # prediction
            hidden = tf.layers.dropout(
                hidden, self.params.dropout, training=self.training)
            self.logits = tf.layers.dense(hidden, 2, None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.prediction = tf.argmax(self.logits, 1, output_type=tf.int32)
            self.probability = tf.nn.softmax(self.logits, 1)

            # choose in all endings
            probability = tf.reshape(
                self.probability, [batch_size, endings_per_context, 2])
            self.prefer = tf.argmax(
                probability[:, :, 1], 1, output_type=tf.int32)

            # training loss
            labels = tf.reshape(self.input_y, [-1])
            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=self.logits)
            self.loss = tf.reduce_mean(log_prob)

            # prediction accuracy
            correct_mask = tf.cast(
                tf.equal(self.prediction, labels), tf.float32)
            self.accuracy = tf.reduce_mean(correct_mask)

        # train op
        global_step = tf.get_variable(
            'global_step', initializer=tf.constant(0), trainable=False)
        optimizer = tf.train.AdamOptimizer()
        grads_and_vars = optimizer.compute_gradients(self.loss)
        tgrads, tvars = zip(*grads_and_vars)
        tgrads, _ = tf.clip_by_global_norm(tgrads, self.params.max_grad_norm)
        grads_and_vars = zip(tgrads, tvars)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step)


def batch_generator(corpus, params, fmt, negative_choices=None):
    '''Batch generator.

    Args:
        corpus: train/val corpus
        params: experiment parameters
        fmt: 'train'/'val'/'test'/'submit'
        negative_choices: set of negative endings
    '''

    assert fmt in ['train', 'val', 'test',
                   'submit'], 'Unexpected value %s for fmt' % fmt

    if fmt == 'train': # train corpus
        X = corpus['story']
        n = X.shape[0]
        permut_idx = np.random.permutation(n)
        n_batch = (n-1)//params.batch_size + 1

        for batch in range(n_batch):
            start = batch * params.batch_size
            end = start + params.batch_size
            idx = permut_idx[start:end] # permuted
            batch_size = idx.shape[0]

            # context
            context = X[idx, :CONTEXT_LENGTH]
            context = context[:, :, 1:] # remove BOS

            # endings
            correct_endings = X[idx, CONTEXT_LENGTH][None, :, :]
            num_negatives = batch_size * params.negative_sampling
            if negative_choices is None:
                negative_endings = X[np.random.randint(
                    n, size=num_negatives), CONTEXT_LENGTH]
            else:
                m = negative_choices.shape[0]
                negative_endings = negative_choices[np.random.randint(
                    m, size=num_negatives)]
            negative_endings = np.reshape(
                negative_endings, (params.negative_sampling, batch_size, -1))
            endings = np.concatenate(
                (correct_endings, negative_endings), axis=0)
            endings = np.transpose(endings, [1, 0, 2])
            endings = endings[:, :, 1:] # remove BOS

            # labels
            labels = np.zeros((batch_size, 1+params.negative_sampling), np.int32)
            labels[:, 0] = 1 # first column is correct ending

            yield context, endings, labels
    elif fmt in ['val', 'test']: # validation/test corpus
        X = corpus['context']
        n = X.shape[0]
        n_batch = (n-1)//params.batch_size + 1
        all_idx = np.arange(n)

        # convert answer to numpy array
        answer = np.array(corpus['answer'], dtype=np.int32)

        if fmt == 'val': # validation corpus (for training)
            for batch in range(n_batch):
                start = batch * params.batch_size
                end = start + params.batch_size
                idx = all_idx[start:end]
                batch_size = idx.shape[0]

                # context
                context = X[idx]
                context = context[:, :, 1:] # remove BOS

                # endings
                endings = corpus['endings'][idx]
                endings = endings[:, :, 1:] # remove BOS

                # labels
                labels = np.zeros((batch_size, endings.shape[1]), np.int32)
                labels[np.arange(batch_size), answer[idx]] = 1

                yield context, endings, labels
        else: # test corpus (for evaludation)
            context = corpus['context'][:, :, 1:] # remove BOS
            endings = corpus['endings'][:, :, 1:] # remove BOS
            labels = np.zeros((n, endings.shape[1]), np.int32)
            labels[np.arange(n), answer] = 1
            yield context, endings, labels
    else: # submit corpus
        context = corpus['context'][:, :, 1:] # remove BOS
        endings = corpus['endings'][:, :, 1:] # remove BOS
        yield context, endings


def train_step(sess, batch, model):
    '''Make a single train step.

    Args:
        sess: Tensorflow session
        batch: training batch
        model: cloze classifier
    '''

    logger = logging.getLogger('__main__')

    context, endings, labels = batch
    feed_dict = {
        model.context: context,
        model.endings: endings,
        model.input_y: labels,
        model.training: True
    }
    train_op = model.train_op
    global_step = tf.train.get_global_step()
    _, step, loss, accuracy = sess.run(
        [train_op, global_step, model.loss, model.accuracy], feed_dict)

    logger.info('step %d, loss %f, accuracy %f' % (step, loss, accuracy))


def val_step(sess, batch, model):
    '''Make a single validation step.

    Args:
        sess: Tensorflow session
        batch: training batch
        model: cloze classifier

    Returns:
        prediction
    '''

    logger = logging.getLogger('__main__')

    context, endings, labels = batch
    feed_dict = {
        model.context: context,
        model.endings: endings,
        model.input_y: labels,
        model.training: False
    }
    loss, accuracy, prediction, prefer = sess.run(
        [model.loss, model.accuracy, model.prediction, model.prefer], feed_dict)

    logger.info('Validation with %d stories' % context.shape[0])

    answer = np.where(labels==1)[1]
    prediction_accuracy = sklearn.metrics.accuracy_score(answer, prefer)
    logger.info('Validation loss %f, clf acc %f, pred acc %f' %
        (loss, accuracy, prediction_accuracy))

    return prediction


def pred_step(sess, batch, model):
    '''Make a single prediction step.

    Args:
        sess: Tensorflow session
        batch: training batch
        model: cloze classifier

    Returns:
        prediction
    '''

    logger = logging.getLogger('__main__')

    context, endings = batch
    feed_dict = {
        model.context: context,
        model.endings: endings,
        model.training: False
    }
    prefer = sess.run(model.prefer, feed_dict)

    logger.info('Prediction with %d stories' % context.shape[0])
    return prefer


def main():
    '''Main function.'''

    # download NLTK resources
    nltk_download()

    # parser for command line arguments
    parser = argparse.ArgumentParser(description='NLU project 2')
    # network architecture
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='Embedding dimension, default 300')
    parser.add_argument('--state_dim', type=int, default=64,
                        help='LSTM cell hidden state dimension ' +
                        '(for c and h), default 64')
    parser.add_argument('--clf_hidden', type=str, default=None,
                        help='Hidden layer dimensions for ending classifier')
    # input data preprocessing
    parser.add_argument('--train_corpus', type=str, default='data/train.csv',
                        help='Path to training corpus')
    parser.add_argument('--val_corpus', type=str, default='data/val.csv',
                        help='Path to evaluation corpus')
    parser.add_argument('--test_corpus', type=str, default='data/test.csv',
                        help='Path to test corpus')
    parser.add_argument('--submit_corpus', type=str, default='data/submit.csv',
                        help='Path to submission corpus')
    parser.add_argument('--prediction', type=str, default=None,
                        help='Path to prediction output file')
    parser.add_argument('--max_sentence_length', type=int, default=40,
                        help='Maximum sentence length to keep, default 40')
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='Vocabulary size, default 20000')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained word embedding, default None')
    parser.add_argument('--max_pretrained_vocab_size', type=int, default=1000000,
                        help='Maximum pretrained tokens to read, default 1000000')
    # training
    parser.add_argument('--dropout', type=float,
                        default=0.75, help='Dropout rate')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch size')
    parser.add_argument('--n_epoch', type=int, default=10,
                        help='Training epoch number, default 10')
    parser.add_argument('--negative_sampling', type=int,
                        default=1, help='Negative sampling, default 1')
    parser.add_argument('--max_grad_norm', type=float,
                        default=5.0, help='Maximum gradient norm')
    # experiment path
    parser.add_argument('--exp_path', type=str, default=None,
                        help='Experiment path')
    params = parser.parse_args()

    # set random seeds
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    # parameter validation
    assert os.path.exists(params.train_corpus)
    assert os.path.exists(params.val_corpus)
    assert os.path.exists(params.test_corpus)
    if params.pretrained is not None:
        assert os.path.exists(params.pretrained)

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

    # load corpora
    train_corpus = load_corpus(params.train_corpus, params, 'train')
    val_corpus = load_corpus(params.val_corpus, params, 'val')
    test_corpus = load_corpus(params.test_corpus, params, 'test')
    submit_corpus = load_corpus(params.submit_corpus, params, 'submit')

    # preprocess corpora
    train_corpus = preprocess_corpus(train_corpus, params, 'train')
    val_corpus = preprocess_corpus(val_corpus, params, 'val')
    test_corpus = preprocess_corpus(test_corpus, params, 'test')
    submit_corpus = preprocess_corpus(submit_corpus, params, 'submit')

    # build dictionary
    dico = build_dictionary(train_corpus, params)
    params.dico = dico
    params.dico_size = len(dico)

    # transform corpora
    train_corpus = transform_corpus(train_corpus, dico, params, 'train')
    val_corpus = transform_corpus(val_corpus, dico, params, 'val')
    test_corpus = transform_corpus(test_corpus, dico, params, 'test')
    submit_corpus = transform_corpus(submit_corpus, dico, params, 'submit')

    # classifier
    logger.info('Building model')
    model = ClozeClassifier(params)

    # train/val split
    n_train = int(0.9 * val_corpus['context'].shape[0])
    dev_corpus = {
        'context': val_corpus['context'][n_train:],
        'endings': val_corpus['endings'][n_train:],
        'answer': val_corpus['answer'][n_train:]
    }
    val_corpus = {
        'context': val_corpus['context'][:n_train],
        'endings': val_corpus['endings'][:n_train],
        'answer': val_corpus['answer'][:n_train]
    }

    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, 1+params.n_epoch):
            logger.info('Start of epoch #%d' % epoch)
            # for batch in batch_generator(train_corpus, params,
            #     'train', val_corpus['endings'][:, 1]):
            #     train_step(sess, batch, model)
            for batch in batch_generator(val_corpus, params, 'val'):
                train_step(sess, batch, model)

            logger.info('Validation on val set')
            for batch in batch_generator(val_corpus, params, 'test'):
                val_step(sess, batch, model)

            logger.info('Validation on dev set')
            for batch in batch_generator(dev_corpus, params, 'test'):
                val_step(sess, batch, model)

            logger.info('Validation on test set')
            for batch in batch_generator(test_corpus, params, 'test'):
                val_step(sess, batch, model)

        # final prediction
        for batch in batch_generator(submit_corpus, params, 'submit'):
            predictions = pred_step(sess, batch, model)
            predictions += 1

        # output predictions
        if params.prediction is not None:
            logger.info('Writing prediction results to %s' % params.prediction)
            with open(params.prediction, 'w') as f:
                for i in predictions:
                    f.write('%d\n' % i)


if __name__ == '__main__':
    main()
