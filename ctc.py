# -*- coding: utf-8 -*-

import numpy as np
import os
import pickle
import soundfile as sf
import sys
import tensorflow as tf

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional, Dense, Activation, TimeDistributed, Lambda, Input 
from base.utils import KmRNNTUtil as util
from base.common import Logger
from base.data_generator import AudioGenerator

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def add_ctc_loss(input_to_softmax):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
    # output_length = BatchNormalization()(input_lengths)
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [input_to_softmax.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths],
        outputs=loss_out)
    return model

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    logger = Logger(name = "KmRNNT", level = Logger.DEBUG).logger

    logger.info("Tensorflow version: %s"%tf.__version__)

    basepath='/home/sephiroce/data/LDC/wsj'

    minibatch_size=80
    feat_dim=40
    optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    epochs=2
    sort_by_duration=False
    max_duration=20.0
    save_model_path="tmp_ctc.ckpt"
    pickle_path = "tmp_ctc.pkl"
    vocab,_ = util.load_vocab('/home/sephiroce/data/LDC/wsj/Processed/bpes/wsj.1k.vocab')
    output_dim = len(vocab)
    cell_size=256

    # create a class instance for obtaining batches of data
    audio_gen = AudioGenerator(logger, basepath = basepath, vocab=vocab,minibatch_size=minibatch_size,
            feat_dim=feat_dim, max_duration=max_duration,
            sort_by_duration=sort_by_duration)

    # add the training data to the generator
    audio_gen.load_train_data('train_corpus.json')
    audio_gen.load_validation_data('valid_corpus.json')

    # calculate steps_per_epoch
    num_train_examples=len(audio_gen.train_audio_paths)
    steps_per_epoch = num_train_examples//minibatch_size

    # calculate validation_steps
    num_valid_samples = len(audio_gen.valid_audio_paths)
    validation_steps = num_valid_samples//minibatch_size

    # Bidirectional CTC
    input_data = Input(name='the_input', shape=(None, feat_dim))
    bidir_rnn = Bidirectional(LSTM(cell_size, return_sequences=True, activation='relu'), merge_mode='concat')(input_data)
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)

    input_to_softmax = Model(inputs=input_data, outputs=y_pred)
    input_to_softmax.output_length = lambda x: x
    input_to_softmax.summary()

    # add CTC loss to the NN specified in input_to_softmax
    model = add_ctc_loss(input_to_softmax)

    # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    # make results/ directory, if necessary
    if not os.path.exists('results'):
        os.makedirs('results')

    # add checkpointer
    checkpointer = ModelCheckpoint(filepath='results/'+save_model_path, verbose=0)

    # train the model
    hist = model.fit_generator(generator = audio_gen.next_train(), 
                               steps_per_epoch = steps_per_epoch,
                               epochs=epochs, 
                               validation_data = audio_gen.next_valid(), 
                               validation_steps = validation_steps,
                               callbacks=[checkpointer], 
                               verbose=1)

    # save model loss
    with open('results/'+pickle_path, 'wb') as f:
        pickle.dump(hist.history, f)

if __name__ == "__main__":
    main()
