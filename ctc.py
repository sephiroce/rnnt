# -*- coding: utf-8 -*-
#pylint: disable=too-many-locals, no-member

"""ctc.py: Building CTC models for ASR tasks."""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import os
import pickle
import sys

from keras.optimizers import SGD
from keras.initializers import RandomUniform
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, model_from_json
from keras.layers import CuDNNLSTM, \
  Bidirectional, Dense, Activation, TimeDistributed, Lambda, Input

from base.common import Constants, Logger, ParseOption, ExitCode
from base.utils import KmRNNTUtil as Util
from base.data_generator import AudioGenerator

class KMCTC:
  @staticmethod
  def get_result_str(utt, id_to_word, is_char=False):
    sent = ""
    for chars in utt:
      if is_char:
        for char in chars:
          if char < 0:
            break
          if id_to_word[int(char)] == Constants.SPACE:
            sent += " "
          else:
            sent += id_to_word[int(char)]
      else:
        for word in chars:
          if word < 0:
            break
          sent += id_to_word[int(word)] +" "
    return sent.strip()

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
  def loading_model(infer_json, infer_h5, train_json, train_h5, config):
    with open(infer_json) as json_file:
      model_4_decoding = model_from_json(json_file.read())
      model_4_decoding.load_weights(infer_h5)
    with open(train_json) as json_file:
      model_4_training = model_from_json(json_file.read())
      model_4_training.load_weights(train_h5)

    return KMCTC.compile_models(model_4_training, model_4_decoding, config)

  @staticmethod
  def compile_models(model_4_training, model_4_decoding, config):
    # multi gpu
    if config.device_number_of_gpu >= 2:
      from keras.utils.training_utils import multi_gpu_model
      model_4_training = multi_gpu_model(model_4_training,
                                         gpus=config.device_number_of_gpu)

    # compiling a model
    optimizer = SGD(lr=config.train_learning_rate,
                    decay=config.train_learning_rate_decay,
                    momentum=config.train_learning_rate_momentum,
                    nesterov=config.train_is_nesterov,
                    clipnorm=config.train_clipping_norm)
    model_4_training.compile(loss={Constants.KEY_CTCLS:
                                   lambda y_true, y_pred: y_pred},
                             optimizer=optimizer)
    model_4_decoding.compile(loss={Constants.KEY_CTCDE:
                                   lambda y_true, y_pred: y_pred},
                             optimizer=optimizer)

    return model_4_training, model_4_decoding

  @staticmethod
  def create_model(logger, config, vocab):
    """
    this method creates a ctc model
    please modify this method directly just like setting hyper-parameters for
    the models.
    :param logger:
    :param vocab:
    :param config:
    :return: a ctc model
    """

    input_dim = config.feature_dimension
    output_dim = len(vocab) + 1
    cell_size = config.model_layer_size
    layers = config.model_number_of_layer

    logger.info("Input dim: %d", input_dim)
    logger.info("Output dim: %d (= vocab size + 1)", output_dim)

    # Bidirectional CTC
    input_data = Input(name=Constants.KEY_INPUT, shape=(None, input_dim))

    prev_layer = input_data
    for _ in range(layers):
      if config.model_rnn_direction == Constants.BI_DIRECTION:
        prev_layer = Bidirectional(CuDNNLSTM(cell_size,
                                             return_sequences=True,
                                             bias_initializer=RandomUniform(
                                                 -config.model_init_scale,
                                                 config.model_init_scale),
                                             kernel_initializer=RandomUniform(
                                                 -config.model_init_scale,
                                                 config.model_init_scale),
                                             recurrent_initializer=RandomUniform(
                                                 -config.model_init_scale,
                                                 config.model_init_scale)),
                                   merge_mode='concat')(prev_layer)
      else:
        prev_layer = CuDNNLSTM(cell_size, return_sequences=True)(prev_layer)

    dense_layer = Dense(output_dim,
                        kernel_initializer=\
                          RandomUniform(minval=-config.model_init_scale,
                                        maxval=config.model_init_scale),
                        bias_initializer=\
                          RandomUniform(minval=-config.model_init_scale,
                                        maxval=config.model_init_scale))

    # Three inputs for
    time_dense = TimeDistributed(dense_layer)(prev_layer) #[Batch, SeqLen, Vocab_N]
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # add CTC loss to the model
    label = Input(name=Constants.KEY_LABEL, shape=[None])
    input_length = Input(name=Constants.KEY_INLEN, shape=[1])
    label_length = Input(name=Constants.KEY_LBLEN, shape=[1])

    loss_out = Lambda(Util.ctc_lambda_func, output_shape=(1,),
                      name=Constants.KEY_CTCLS)\
      ([y_pred] + [label, input_length, label_length])

    # Setting decoder
    out_decoded_dense = Lambda(KMCTC.ctc_complete_decoding_lambda_func,
                               output_shape=(None, None),
                               name=Constants.KEY_CTCDE,
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

    return KMCTC.compile_models(model_4_training, model_4_decoding, config)

def main():
  logger = Logger(name="KmCTC", level=Logger.DEBUG).logger

  # Configurations
  config = ParseOption(sys.argv, logger).args

  model_name = "%s%s%d.layer%d_%d_lr%f_gpus%d" % (config.model_rnn_direction,
                                                  config.feature_type,
                                                  config.feature_dimension,
                                                  config.model_number_of_layer,
                                                  config.model_layer_size,
                                                  config.train_learning_rate,
                                                  config.device_number_of_gpu)
  logger.info("Model name: %s", model_name)

  # Loading vocabs
  vocab_path = Util.get_file_path(config.paths_data_path, config.paths_vocab)
  if not os.path.isfile(vocab_path):
    logger.critical("%s does not exist.", vocab_path)
    sys.exit(ExitCode.INVALID_FILE_PATH)
  vocab, _ = Util.load_vocab(vocab_path, config=config)
  logger.info("%d words were loaded.", len(vocab))

  # paths for saving models
  infer_json = "%s/checkpoints/%s_inference.json"%(config.paths_data_path,
                                                   model_name)
  infer_h5 = "%s/checkpoints/%s_inference.h5"%(config.paths_data_path,
                                               model_name)
  train_json = "%s/checkpoints/%s_training.json" % (config.paths_data_path,
                                                    model_name)
  train_h5 = "%s/checkpoints/%s_training.h5" % (config.paths_data_path,
                                                model_name)

  if os.path.isfile(infer_json) and \
    os.path.isfile(infer_h5) and \
    os.path.isfile(train_json) and \
    os.path.isfile(train_h5):
    model_4_training, model_4_decoding = \
      KMCTC.loading_model(infer_json=infer_json,
                          infer_h5=infer_h5,
                          train_json=train_json,
                          train_h5=train_h5,
                          config=config)
    logger.info("A model was loaded.")
  else:
    model_4_training, model_4_decoding = \
      KMCTC.create_model(logger, config, vocab)
    logger.info("A model was created.")

  model_4_training.summary()

  # make checkpoints/ directory, if necessary
  if not os.path.exists(config.paths_data_path + '/checkpoints'):
    os.makedirs(config.paths_data_path + '/checkpoints')

  # saving the meta files of models
  with open(infer_json, "w") as json_file:
    json_file.write(model_4_decoding.to_json())
    logger.info("Saved a meta file of an inference model to %s", infer_json)

  with open(train_json, "w") as json_file:
    json_file.write(model_4_training.to_json())
    logger.info("Saved a meta file of a training model to %s", train_json)

  # create a class instance for obtaining batches of data
  audio_gen = AudioGenerator(logger, config, vocab)

  # add the training data to the generator
  audio_gen.load_train_data(Util.get_file_path(config.paths_data_path,
                                               "train_corpus.json"))
  audio_gen.load_validation_data(Util.get_file_path(config.paths_data_path,
                                                    "valid_corpus.json"))
  audio_gen.load_test_data(Util.get_file_path(config.paths_data_path,
                                              "test_corpus.json"))

  # train a new model
  train_batch_size = (len(audio_gen.train_audio_paths)//config.train_batch)
  valid_batch_size = (len(audio_gen.valid_audio_paths)//config.train_batch)

  # call back functions
  early_stopping = EarlyStopping(monitor=Constants.VAL_LOSS, patience=200,
                                 verbose=1,
                                 mode='min')

  checkpoint = ModelCheckpoint('%s/checkpoints/%s-{epoch:03d}-{loss:03f}-{'
                               'val_loss:03f}.h5'
                               %(config.paths_data_path, model_name),
                               verbose=1, monitor=Constants.VAL_LOSS,
                               save_best_only=False, mode='auto')

  # fitting a model
  hist = model_4_training.fit_generator(generator=audio_gen.next_train(),
                                        steps_per_epoch=train_batch_size,
                                        epochs=config.train_max_epoch,
                                        validation_data=audio_gen.next_valid(),
                                        validation_steps=valid_batch_size,
                                        callbacks=[early_stopping, checkpoint],
                                        verbose=1)
  model_4_decoding.set_weights(model_4_training.get_weights())

  # saving model weights
  model_4_decoding.save_weights(infer_h5)
  logger.info("Saved weights of the inference model to %s", infer_h5)
  model_4_training.save_weights(train_h5)
  logger.info("Saved weights of the training model to %s", train_h5)

  # saving pickle_path
  with open("%s/checkpoints/%s_ctc_loss.pkl"%(config.paths_data_path,
                                              model_name), 'wb') as file:
    pickle.dump(hist.history, file)

if __name__ == "__main__":
  main()
