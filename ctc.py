# -*- coding: utf-8 -*-
#pylint: disable=too-many-locals,

"""ctc.py: Building CTC models for ASR tasks."""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import os
import pickle
import sys

from keras.optimizers import SGD
from keras.models import Model
from keras.layers import CuDNNLSTM, Bidirectional, Dense, Activation, TimeDistributed, Lambda, Input
from base.common import Constants
from base.utils import KmRNNTUtil as Util
from base.common import Logger
from base.data_generator import AudioGenerator

# Setting hyper-parameters
epochs = 10
minibatch_size = 80
sort_by_duration = False
max_duration = 50.0
is_char = True
is_bos_eos = False

# Feature
feat_dim = 40

# Model architecture
cell_size = 300
optimizer = SGD(lr=1e-4, decay=1e-5, momentum=0.9, nesterov=True, clipnorm=5)
n_gpu = 1

# Paths
basepath = sys.argv[1]
model_4_decoding_json = "inference.new.json"
model_4_decoding_h5 = "inference.new.h5"
pickle_path = "tmp_ctc.new.pkl"

class KMCTC:
  @staticmethod
  def print_result(logger, utts, id_to_word):
    for i, utt in enumerate(utts):
      sent = ""
      for char in utt:
        if char < 0:
          break
        if int(char) == len(id_to_word) - 1:
          sent += " "
        else:
          sent += id_to_word[int(char)]
      logger.info("%d: %s" % (i, sent))

  @staticmethod
  def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    shift = 2
    y_pred = y_pred[:, shift:, :]
    input_length -= shift
    from keras import backend as k
    return k.ctc_batch_cost(labels, y_pred, input_length, label_length)

  @staticmethod
  def ctc_complete_decoding_lambda_func(args, **arguments):
    """
    borrowed from https://github.com/ysoullard/CTCModel/blob/master/CTCModel.py
    :param args:
    :param arguments:
    :return:
    """
    y_pred, input_length = args
    my_params = arguments
    from keras import backend as k
    return k.cast(k.ctc_decode(y_pred, k.squeeze(input_length, 1),
                               greedy=my_params['greedy'],
                               beam_width=my_params['beam_width'],
                               top_paths=my_params['top_paths'])[0][0],
                  dtype='float32')

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
    input_data = Input(name=Constants.KEY_INPUT, shape=(None, input_dim))

    blstm1 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True),
                           merge_mode='concat')(input_data)
    blstm2 = Bidirectional(CuDNNLSTM(cell_size, return_sequences=True),
                           merge_mode='concat')(blstm1)
    time_dense = TimeDistributed(Dense(output_dim + 1))(blstm2)
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # add CTC loss to the model
    label = Input(name=Constants.KEY_LABEL, shape=[None])
    input_length = Input(name=Constants.KEY_INLEN, shape=[1])
    label_length = Input(name=Constants.KEY_LBLEN, shape=[1])

    loss_out = Lambda(KMCTC.ctc_lambda_func, output_shape=(1,), name=Constants.KEY_CTCLS)\
      ([y_pred] + [label, input_length, label_length])

    out_decoded_dense = Lambda(KMCTC.ctc_complete_decoding_lambda_func,
                               output_shape=(None, None), name='CTCdecode',
                               arguments={'greedy': False,
                                          'beam_width': 12,
                                          'top_paths': 1},
                               dtype="float32")\
      ([y_pred] + [input_length])

    # creating a model
    model = Model(inputs=[input_data] + [label, input_length, label_length],
                  outputs=loss_out)
    model_4_decoding = Model(inputs=[input_data] + [input_length],
                             outputs=out_decoded_dense)

    # multi gpu
    if gpus >= 2:
      from keras.utils.training_utils import multi_gpu_model
      model = multi_gpu_model(model, gpus=gpus)

    # compiling a model
    model.compile(loss={Constants.KEY_CTCLS: lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    model_4_decoding.compile(loss={'CTCdecode': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    return model_4_decoding, model

def main():
  logger = Logger(name="KmRNNT", level=Logger.DEBUG).logger
  vocab, id_to_word = Util.load_vocab(sys.argv[2], is_char=is_char,
                                      is_bos_eos=is_bos_eos)
  logger.info("The number of vocabularies is %d", len(vocab))

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
  model_4_decoding.set_weights(model_4_training.get_weights())

  # saving model
  infer_json = "results/%s"%model_4_decoding_json
  infer_h5 = "results/%s"%model_4_decoding_h5
  model_json = model_4_decoding.to_json()
  with open(infer_json, "w") as json_file:
    json_file.write(model_json)
  model_4_decoding.save_weights(infer_h5)
  logger.info("Saved to %s, %s", infer_json, infer_h5)

  utts = model_4_decoding.predict_generator(generator=audio_gen.next_test(),
                                            steps=1, verbose=1)
  KMCTC.print_result(logger, utts, id_to_word)

  with open('results/'+pickle_path, 'wb') as file:
    pickle.dump(hist.history, file)

if __name__ == "__main__":
  main()
