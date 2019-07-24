# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals,

import os
import sys

from keras import backend as k
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import LSTM, Bidirectional, Dense, Activation, TimeDistributed, Lambda, Input
from base.utils import KmRNNTUtil as Util
from base.common import Logger
from base.data_generator import AudioGenerator
import keras
import numpy as np

def ctc_lambda_func(args):
  y_pred, labels, input_length, label_length = args
  return k.ctc_batch_cost(labels, y_pred, input_length, label_length)

def add_ctc_loss(input_to_softmax):
  the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
  input_lengths = Input(name='input_length', shape=(1,), dtype='int32')
  label_lengths = Input(name='label_length', shape=(1,), dtype='int32')
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
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  feat_dim = 20
  logger = Logger(name="KmRNNT_CTC_Decoder", level=Logger.DEBUG).logger
  vocab, v = Util.load_vocab(sys.argv[2], is_char=True, is_bos_eos=False)
  output_dim = len(vocab)
  cell_size= 300

  input_data = Input(name='the_input', shape=(None, feat_dim))
  # Add convolutional layer
  """
  conv_1d = Conv1D(filters, kernel_size,
                   strides=conv_stride,
                   padding=conv_border_mode,
                   activation='relu',
                   name='conv1d')(input_data)
  # Add batch normalization
  bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
  """
  bidir_rnn = Bidirectional(LSTM(cell_size, return_sequences=True,
                                 activation='relu'), merge_mode='concat')\
    (input_data)
  time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
  y_pred = Activation('softmax', name='softmax')(time_dense)

  input_to_softmax = Model(inputs=input_data, outputs=y_pred)
  input_to_softmax.output_length = lambda x: x
  input_to_softmax.summary()

  # add CTC loss to the NN specified in input_to_softmax
  model = add_ctc_loss(input_to_softmax)

  # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
  optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
  model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
  model.load_weights("results/tmp_ctc.ckpt")
  model.summary()

  vocab, _ = Util.load_vocab(sys.argv[2], is_char=True, is_bos_eos=False)
  eps = 1e-14

  # Get Features
  feature = Util.get_logfbank(sys.argv[1],feat_dim)
  feats_mean = np.load("/home/sephiroce/data/LDC/wsj/wsj.mean")
  feats_std = np.load("/home/sephiroce/data/LDC/wsj/wsj.vari")
#  feature = (feature - feats_mean) / (feats_std + eps)
  feature = np.expand_dims(feature,axis=0)
  print(feature)
  # Batch decoding
  predict=input_to_softmax.predict(feature)
  print(predict)
  print(len(predict[0]))
  for vec in predict[0]:
    if np.argmax(vec) != 29:
      print(np.argmax(vec))

"""
  line = ""
  for i in range(1):
    for word in decoded_sequences[0][i]:
      line += v[word]
#    print(line.replace("<space>", " "))
    print("["+line+"]")
"""

if __name__ == "__main__":
  main()
