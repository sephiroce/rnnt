# -*- coding: utf-8 -*-
# pylint: disable=no-member, import-error, no-name-in-module, too-many-arguments
# pylint: disable=too-many-locals

"""keras_model.py: Seq2Seq model based on Keras"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import sys

from keras.initializers import RandomUniform
from keras.layers import Bidirectional, Input, Lambda, Activation
from keras.layers import Dense, TimeDistributed, Dropout, GaussianNoise
from keras.layers import CuDNNLSTM as LSTM
from keras.models import Model
from keras.optimizers import SGD

from rnnt.base.common import Constants, ExitCode, OutputType
from rnnt.base.util import Util

class KerasModel(object):
  @staticmethod
  def get_rnn(input_vector, output_dim, num_hidden, num_layers, dropout,
              is_bidirectional, output_layer_name, gaussian_noise=0.0,
              init_range=0.1, output_type=OutputType.LOGIT):
    prev_layer = None

    for _ in range(num_layers):
      if prev_layer is None:
        prev_layer = input_vector

        # Adding gaussian noise only for the input data
        if gaussian_noise > 0.0:
          prev_layer = GaussianNoise(gaussian_noise)(prev_layer)

      lstm_layer = LSTM(num_hidden, return_sequences=True,
                        bias_initializer=\
                          RandomUniform(minval=-init_range,
                                        maxval=init_range),
                        kernel_initializer=\
                          RandomUniform(minval=-init_range,
                                        maxval=init_range),
                        recurrent_initializer=\
                          RandomUniform(minval=-init_range,
                                        maxval=init_range))

      if is_bidirectional:
        prev_layer = Bidirectional(lstm_layer, merge_mode='concat')(prev_layer)
      else:
        prev_layer = lstm_layer(prev_layer)

      if dropout > 0:
        prev_layer = Dropout(dropout)(prev_layer)

    if output_type == OutputType.HIDDEN:
      return Model(inputs=[input_vector], outputs=prev_layer)

    time_dense = TimeDistributed(Dense(output_dim,
                                       bias_initializer= \
                                         RandomUniform(minval=-init_range,
                                                       maxval=init_range),
                                       kernel_initializer= \
                                         RandomUniform(minval=-init_range,
                                                       maxval=init_range)),
                                 name=None if output_type == OutputType.SOFTMAX
                                 else output_layer_name)(prev_layer)

    if output_type == OutputType.SOFTMAX:
      output = Activation('softmax', name=output_layer_name)(time_dense)
    else:
      output = time_dense

    return Model(inputs=[input_vector], outputs=output)

  @staticmethod
  def get_fc(input_vector, output_dim, num_hidden, num_layers, dropout,
             init_range):
    assert num_layers > 0 and num_hidden > 0 and output_dim > 0
    prev_layer = None
    for layer_i in range(num_layers):
      prev_layer = \
        TimeDistributed(Dense(output_dim if layer_i == num_layers - 1 else
                              num_hidden,
                              activation="linear" if layer_i == num_layers - 1
                              else "tanh",
                              bias_initializer=\
                                RandomUniform(minval=-init_range,
                                              maxval=init_range),
                              kernel_initializer=\
                                RandomUniform(minval=-init_range,
                                              maxval=init_range)))\
        (input_vector if prev_layer is None else prev_layer)
      if dropout > 0:
        prev_layer = Dropout(rate=dropout)(prev_layer)
    return prev_layer

  @staticmethod
  def create_model(config, vocab, model_type):
    # Inputs
    input_tran = \
      Input(name=Constants.INPUT_TRANS, shape=[None, config.feature_dimension])
    input_length = Input(name=Constants.INPUT_INLEN, shape=[1], dtype='int32')
    label_length = Input(name=Constants.INPUT_LBLEN, shape=[1], dtype='int32')
    label = Input(name=Constants.INPUT_LABEL, shape=[None], dtype='int32')

    # CTC model: Bidirectional LSTM
    if model_type == Constants.CTC:
      output_type = OutputType.SOFTMAX
    elif model_type == Constants.RNNT:
      output_type = OutputType.LOGIT
    elif model_type == Constants.RNNT_FF:
      output_type = OutputType.HIDDEN
    else:
      raise ExitCode.INVALID_OPTION

    encoder = KerasModel.get_rnn(input_tran,
                                 len(vocab) + 1,
                                 config.encoder_layer_size,
                                 config.encoder_number_of_layer,
                                 config.encoder_dropout,
                                 config.encoder_rnn_direction == 'bi',
                                 Constants.OUTPUT_TRANS,
                                 config.train_gaussian_noise,
                                 config.model_init_scale,
                                 output_type=output_type)

    is_rnnt = model_type == Constants.RNNT or model_type == Constants.RNNT_FF
    if is_rnnt:
      input_pred = Input(name=Constants.INPUT_PREDS, shape=[None, len(vocab)])

      decoder = KerasModel.get_rnn(input_pred,
                                   len(vocab) + 1,
                                   config.decoder_layer_size,
                                   config.decoder_number_of_layer,
                                   config.decoder_dropout,
                                   False,
                                   Constants.OUTPUT_PREDS,
                                   0.0,
                                   config.model_init_scale,
                                   output_type=output_type)

      if model_type == Constants.RNNT:
        loss_out = Lambda(Util.rnnt_lambda_func, output_shape=(1,),
                          name=Constants.LOSS_RNNT) \
          ([encoder.output, decoder.output, label, input_length, label_length])
      elif model_type == Constants.RNNT_FF:
        # Joint encoder and decoder
        joint_pred = Lambda(Util.concatenate_lambda,
                            output_shape=[None, None, config.encoder_layer_size \
                                          * 2 + config.decoder_layer_size],
                            name="joint")([encoder.output, decoder.output])

        # Fully Connected network
        acts = KerasModel.get_fc(joint_pred,
                                 len(vocab) + 1,
                                 config.joint_layer_size,
                                 config.joint_number_of_layer,
                                 config.joint_dropout,
                                 config.model_init_scale)

        loss_out = Lambda(Util.rnnt_lambda_func_v2,
                          output_shape=(1,),
                          name=Constants.LOSS_RNNT) \
          ([acts, label, input_length, label_length])
      else:
        sys.exit(ExitCode.INVALID_CONDITION)

      model = Model(inputs=[input_tran, input_pred, label, input_length,
                            label_length],
                    outputs=loss_out, name="rnnt")
    else:
      loss_out = Lambda(Util.ctc_lambda_func, output_shape=(1,),
                        name=Constants.LOSS_CTC) \
        ([encoder.output] + [label, input_length, label_length])

      model = Model(inputs=[input_tran] + [label, input_length, label_length],
                    outputs=loss_out, name="ctc")

    # multi gpu
    if config.device_number_of_gpu >= 2:
      from keras.utils.training_utils import multi_gpu_model
      model = multi_gpu_model(model, gpus=config.device_number_of_gpu)

    # compiling a model
    optimizer = SGD(lr=config.train_learning_rate,
                    decay=config.train_decay,
                    momentum=config.train_momentum,
                    nesterov=config.train_is_nesterov,
                    clipnorm=config.train_clipping_norm)

    if is_rnnt:
      model.compile(loss={Constants.LOSS_RNNT: lambda y_true, y_pred: y_pred},
                    optimizer=optimizer)
    else:
      model.compile(loss={Constants.LOSS_CTC: lambda y_true, y_pred: y_pred},
                    optimizer=optimizer)

    return model
