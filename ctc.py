# -*- coding: utf-8 -*-
#pylint: disable=too-many-locals,

"""ctc.py: Building CTC models for ASR tasks."""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import os
import pandas as pd
import pickle
import sys
import numpy as np

from keras import backend as k
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Flatten, Conv2D, MaxPooling2D, BatchNormalization, CuDNNLSTM, Bidirectional, Dense, Activation, TimeDistributed, Lambda, Input
from base.utils import KmRNNTUtil as Util
from base.common import Logger
from base.data_generator import AudioGenerator

# Setting hyper-parameters
epochs = 3
minibatch_size = 80
sort_by_duration = False
max_duration = 50.0
is_char = True
is_bos_eos = False

# Feature
feat_dim = 40

# Model architecture
cell_size = 300
optimizer = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
#optimizer = SGD(lr=1e-4, decay=1e-5, momentum=0.0, nesterov=False)
n_gpu = 1

# Paths
basepath = sys.argv[1]
model_4_training_json = "ctc_loss.new.json"
model_4_decoding_json = "inference.new.json"
model_4_training_h5 = "ctc_loss.new.h5"
model_4_decoding_h5 = "inference.new.h5"
pickle_path = "tmp_ctc.new.pkl"

class KMCTC:
  @staticmethod
  def ctc_lambda_func(args):
    labels, y_pred, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    shift = 2
    y_pred = y_pred[:, shift:, :]
    input_length -= shift
    return k.ctc_batch_cost(labels, y_pred, input_length, label_length)

  @staticmethod
  def create_model(input_dim, output_dim, gpus=1):
    """
    this method creates a ctc model
    please modify this method directly just like setting hyper-parameters for the models.
    :param input_dim: feat_dim
    :param output_dim: the number of vocabularies
    :param gpus: the number of gpus
    :return: a ctc model
    """

    # Bidirectional CTC
    input_data = Input(name='the_input', shape=(None, input_dim))

    blstm1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True
                                ), merge_mode='concat')(input_data)
    blstm2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True
                                ), merge_mode='concat')(blstm1)
    time_dense = TimeDistributed(Dense(output_dim + 1))(blstm2)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model_4_decoding = Model(inputs=input_data, outputs=y_pred)

    # add CTC loss to the model
    label = Input(name='the_labels', shape=(None,), dtype='float32')
    input_length = Input(name='input_length', shape=(1,), dtype='int32')
    label_length = Input(name='label_length', shape=(1,), dtype='int32')
    output_length = Lambda(lambda x: x)(input_length)

    loss_out = Lambda(KMCTC.ctc_lambda_func, output_shape=(1,), name='ctc')(
        [label, model_4_decoding.output, output_length, label_length])

    # creating a model
    model = Model(
        inputs=[model_4_decoding.input, label, input_length, label_length],
        outputs=loss_out)

    # multi gpu
    if gpus >= 2:
      from keras.utils.training_utils import multi_gpu_model
      model = multi_gpu_model(model, gpus=gpus)
    
    # compiling a model
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                             optimizer=optimizer)
    return model_4_decoding, model

def main():
  logger = Logger(name="KmRNNT", level=Logger.DEBUG).logger
  vocab, _ = Util.load_vocab(sys.argv[2], is_char=is_char,
                             is_bos_eos=is_bos_eos)
  logger.info("The number of vocabularies is %d"%len(vocab))

  # create a class instance for obtaining batches of data
  audio_gen = AudioGenerator(logger, basepath=basepath, vocab=vocab,
                             minibatch_size=minibatch_size, feat_dim=feat_dim,
                             max_duration=max_duration,
                             sort_by_duration=sort_by_duration,
                             is_char=is_char, is_bos_eos=is_bos_eos)

  # add the training data to the generator
  audio_gen.load_train_data("%s/train_corpus.json"%basepath)
  audio_gen.load_validation_data('%s/valid_corpus.json'%basepath)
  audio_gen.load_test_data("%s/test_corpus.json"%basepath)

  model_4_decoding, model_4_training = \
    KMCTC.create_model(input_dim=feat_dim,
                       output_dim=len(vocab),
                       gpus=n_gpu)
  logger.info("Model Summary")
  model_4_training.summary()

  # make results/ directory, if necessary
  if not os.path.exists('results'):
    os.makedirs('results')

  # train the model
  train_batch_size = (len(audio_gen.train_audio_paths)//minibatch_size)
  valid_batch_size = (len(audio_gen.valid_audio_paths)//minibatch_size)
  hist = model_4_training.fit_generator(generator=audio_gen.next_train(),
                                        steps_per_epoch=train_batch_size,
                                        epochs=epochs,
                                        validation_data=audio_gen.next_valid(),
                                        validation_steps=valid_batch_size,
                                        verbose=1)

  # saving model
  infer_json = "results/%s"%model_4_decoding_json
  infer_h5 = "results/%s"%model_4_decoding_h5
  model_json = model_4_decoding.to_json()
  with open(infer_json, "w") as json_file:
    json_file.write(model_json)
  model_4_decoding.save_weights(infer_h5)
  logger.info("Saved to %s, %s", infer_json, infer_h5)

  y_pred_proba = model_4_decoding.predict_generator(generator=audio_gen.next_test(),
          steps=1, verbose=1)
  for i, pred in enumerate(y_pred_proba):
      np.savetxt("tmp%s.np"%i, pred)

  with open('results/'+pickle_path, 'wb') as file:
    pickle.dump(hist.history, file)

if __name__ == "__main__":
  main()
