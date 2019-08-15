# -*- coding: utf-8 -*-
#pylint: disable=too-many-locals,

"""ctc.py: Building CTC models for ASR tasks."""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import os
import pickle
import sys

from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model, model_from_json
from keras.layers import CuDNNLSTM, Bidirectional, Dense, Activation, TimeDistributed, Lambda, Input
from base.common import Constants
from base.utils import KmRNNTUtil as Util
from base.common import Logger
from base.data_generator import AudioGenerator

# Setting hyper-parameters
layers = -1
cell_size = -1
lr = -1.0
sort_by_duration = False
max_duration = 50.0
is_char = True
is_bos_eos = False

# Feature
feat_dim = 40
feat_type = Constants.FEAT_FBANK

is_big = False
n_gpu = 1

if is_big:
  epochs = 100
  minibatch_size = 50
  layers = 5
  cell_size = 500
  lr = 0.001 * n_gpu
  optimizer = SGD(lr=lr, decay=lr * 0.0001, momentum=0.9, nesterov=True, clipnorm=5)
else:
  epochs = 20
  minibatch_size = 80
  layers = 2
  cell_size = 300
  lr = 0.02 * n_gpu
  optimizer = SGD(lr=lr, decay=lr * 0.0001, momentum=0.9, nesterov=True, clipnorm=5)

# Paths
basepath = sys.argv[1]
model_name = "%s%d.layer%d_%d_lr%f"%(feat_type, feat_dim, layers, cell_size, lr)

class KMCTC:
  @staticmethod
  def get_result_str(utt, id_to_word):
    sent = ""
    for chars in utt:
      for char in chars:
        if char < 0:
          break
        if id_to_word[int(char)] == Constants.SPACE:
          sent += " "
        else:
          sent += id_to_word[int(char)]
    return sent

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
  def loading_model(infer_json, infer_h5, train_json, train_h5, gpus=n_gpu):
    with open(infer_json) as json_file:
      model_4_decoding = model_from_json(json_file.read())
      model_4_decoding.load_weights(infer_h5)
    with open(train_json) as json_file:
      model_4_training = model_from_json(json_file.read())
      model_4_training.load_weights(train_h5)

    return KMCTC.compile_models(model_4_training, model_4_decoding, gpus)

  @staticmethod
  def compile_models(model_4_training, model_4_decoding, gpus):
    # multi gpu
    if gpus >= 2:
      from keras.utils.training_utils import multi_gpu_model
      model_4_training = multi_gpu_model(model_4_training, gpus=gpus)

    # compiling a model
    model_4_training.compile(loss={Constants.KEY_CTCLS:
                                   lambda y_true, y_pred: y_pred},
                             optimizer=optimizer)
    model_4_decoding.compile(loss={Constants.KEY_CTCDE:
                                   lambda y_true, y_pred: y_pred},
                             optimizer=optimizer)

    return model_4_training, model_4_decoding

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

    blstm = list()
    blstm.append(Bidirectional(CuDNNLSTM(cell_size, return_sequences=True),
                               merge_mode='concat')(input_data))
    for _ in range(layers - 1):
      blstm.append(Bidirectional(CuDNNLSTM(cell_size, return_sequences=True),
                                 merge_mode='concat')(blstm[-1]))

    time_dense = TimeDistributed(Dense(output_dim + 1))(blstm[-1])
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # add CTC loss to the model
    label = Input(name=Constants.KEY_LABEL, shape=[None])
    input_length = Input(name=Constants.KEY_INLEN, shape=[1])
    label_length = Input(name=Constants.KEY_LBLEN, shape=[1])

    loss_out = Lambda(KMCTC.ctc_lambda_func, output_shape=(1,), name=Constants.KEY_CTCLS)\
      ([y_pred] + [label, input_length, label_length])

    # Setting decoder
    out_decoded_dense = Lambda(KMCTC.ctc_complete_decoding_lambda_func,
                               output_shape=(None, None), name=Constants.KEY_CTCDE,
                               arguments={'greedy': False,
                                          'beam_width': 12,
                                          'top_paths': 1},
                               dtype="float32")\
      ([y_pred] + [input_length])

    # creating a model
    model_4_training = Model(inputs=[input_data] + [label, input_length, label_length],
                             outputs=loss_out)
    model_4_decoding = Model(inputs=[input_data] + [input_length],
                             outputs=out_decoded_dense)

    return KMCTC.compile_models(model_4_training, model_4_decoding, gpus)

def main():
  logger = Logger(name="KmRNNT", level=Logger.DEBUG).logger
  vocab, _ = Util.load_vocab(sys.argv[2], is_char=is_char,
                             is_bos_eos=is_bos_eos)
  logger.info("The number of vocabularies is %d", len(vocab))

  # create a class instance for obtaining batches of data
  audio_gen = AudioGenerator(logger, basepath=basepath, vocab=vocab,
                             minibatch_size=minibatch_size, feat_dim=feat_dim,
                             feat_type=feat_type,
                             max_duration=max_duration,
                             sort_by_duration=sort_by_duration,
                             is_char=is_char, is_bos_eos=is_bos_eos)

  # add the training data to the generator
  audio_gen.load_train_data("%s/train_corpus.json"%basepath)
  audio_gen.load_validation_data('%s/valid_corpus.json'%basepath)
  audio_gen.load_test_data("%s/test_corpus.json"%basepath)

  # creating or loading a ctc model
  infer_json = "results/%s_inference.json"%model_name
  infer_h5 = "results/%s_inference.h5"%model_name
  train_json = "results/%s_training.json" % model_name
  train_h5 = "results/%s_training.h5" % model_name

  if os.path.isfile(infer_json) and \
    os.path.isfile(infer_h5) and \
    os.path.isfile(train_json) and \
    os.path.isfile(train_h5):
    model_4_training, model_4_decoding = \
      KMCTC.loading_model(infer_json=infer_json,
                          infer_h5=infer_h5,
                          train_json=train_json,
                          train_h5=train_h5,
                          gpus=n_gpu)
    logger.info("A model was loaded.")
  else:
    model_4_training, model_4_decoding = \
      KMCTC.create_model(input_dim=feat_dim,
                         output_dim=len(vocab),
                         gpus=n_gpu)
    logger.info("A model was created.")

  model_4_training.summary()

  # make results/ directory, if necessary
  if not os.path.exists('results'):
    os.makedirs('results')

  # saving the meta files of models
  with open(infer_json, "w") as json_file:
    json_file.write(model_4_decoding.to_json())
    logger.info("Saved a meta file of an inference model to %s", infer_json)

  with open(train_json, "w") as json_file:
    json_file.write(model_4_training.to_json())
    logger.info("Saved a meta file of a training model to %s", train_json)

  # train a new model
  train_batch_size = (len(audio_gen.train_audio_paths)//minibatch_size)
  valid_batch_size = (len(audio_gen.valid_audio_paths)//minibatch_size)

  # call back functions
  early_stopping = EarlyStopping(monitor=Constants.VAL_LOSS, patience=10,
                                 verbose=0,
                                 mode='min')

  checkpoint = ModelCheckpoint('results/%s-{epoch:03d}-{loss:03f}-{val_loss:03f}.h5'
                               %model_name,
                               verbose=1, monitor=Constants.VAL_LOSS,
                               save_best_only=False, mode='auto')

  reduce_lr_loss = ReduceLROnPlateau(monitor=Constants.VAL_LOSS, factor=0.1,
                                     patience=3,
                                     verbose=1, min_delta=1e-4, mode='min')

  # fitting a model
  hist = model_4_training.fit_generator(generator=audio_gen.next_train(),
                                        steps_per_epoch=train_batch_size,
                                        epochs=epochs,
                                        validation_data=audio_gen.next_valid(),
                                        validation_steps=valid_batch_size,
                                        callbacks=[early_stopping, checkpoint, reduce_lr_loss],
                                        verbose=1)
  model_4_decoding.set_weights(model_4_training.get_weights())


  # saving model weights
  model_4_decoding.save_weights(infer_h5)
  logger.info("Saved weights of the inference model to %s", infer_h5)
  model_4_training.save_weights(train_h5)
  logger.info("Saved weights of the training model to %s", train_h5)

  # saving pickle_path
  with open("results/%s_ctc_loss.pkl"%model_name, 'wb') as file:
    pickle.dump(hist.history, file)

if __name__ == "__main__":
  main()
