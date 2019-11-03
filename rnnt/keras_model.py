# -*- coding: utf-8 -*-
# pylint: disable=no-member, import-error, no-name-in-module, too-many-arguments
# pylint: disable=too-many-locals

"""keras_model.py: Seq2Seq model based on Keras"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

from keras.initializers import RandomUniform
from keras.layers import Bidirectional, Input, Lambda, Activation
from keras.layers import Dense, TimeDistributed, Dropout, GaussianNoise
from keras.layers import CuDNNLSTM as LSTM
from keras.models import Model
from keras.optimizers import SGD

from rnnt.base.common import Constants
from rnnt.base.util import Util


class KerasModel(object):
  @staticmethod
  def get_rnn(input_vector, output_dim, num_hidden, num_layers, dropout,
              is_bidirectional, output_layer_name,
              gaussian_noise=0.0, init_range=0.1, is_softmax=True):
    prev_layer = None

    for _ in range(num_layers):
      if not prev_layer:
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

    time_dense = TimeDistributed(Dense(output_dim,
                                       bias_initializer= \
                                         RandomUniform(minval=-init_range,
                                                       maxval=init_range),
                                       kernel_initializer= \
                                         RandomUniform(minval=-init_range,
                                                       maxval=init_range)),
                                 name=None if is_softmax else output_layer_name)\
      (prev_layer)

    if is_softmax:
      output = Activation('softmax', name=output_layer_name)(time_dense)
    else:
      output = time_dense

    return Model(inputs=[input_vector], outputs=output)

  @staticmethod
  def create_model(config, vocab, is_rnnt):
    # Inputs
    input_tran = Input(name=Constants.INPUT_TRANS, shape=[None, config.feature_dimension])
    input_length = Input(name=Constants.INPUT_INLEN, shape=[1], dtype='int32')
    label_length = Input(name=Constants.INPUT_LBLEN, shape=[1], dtype='int32')
    label = Input(name=Constants.INPUT_LABEL, shape=[None], dtype='int32')

    # CTC model: Bidirectional LSTM
    encoder = KerasModel.get_rnn(input_tran,
                                 len(vocab) + 1,
                                 config.encoder_layer_size,
                                 config.encoder_number_of_layer,
                                 config.encoder_dropout,
                                 config.encoder_rnn_direction == 'bi',
                                 Constants.OUTPUT_TRANS,
                                 config.train_gaussian_noise,
                                 config.model_init_scale,
                                 is_softmax=not is_rnnt)

    if is_rnnt:
      input_pred = Input(name=Constants.INPUT_PREDS, shape=[None, len(vocab)])

      # RNN Language Model: Unidirectional LSTM
      decoder = KerasModel.get_rnn(input_pred,
                                   len(vocab) + 1,
                                   config.decoder_layer_size,
                                   config.decoder_number_of_layer,
                                   config.decoder_dropout,
                                   False,
                                   Constants.OUTPUT_PREDS,
                                   0.0,
                                   config.model_init_scale,
                                   is_softmax=False)

      loss_out = Lambda(Util.rnnt_lambda_func, output_shape=(1,),
                        name=Constants.LOSS_RNNT) \
        ([encoder.output, decoder.output, label, input_length, label_length])

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
