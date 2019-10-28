# -*- coding: utf-8 -*-
# pylint: disable=no-member, import-error, no-name-in-module, too-many-arguments
import os
import pickle
import shutil
import sys

from keras.layers import Bidirectional
from keras.layers import Dense, TimeDistributed, Input, CuDNNLSTM, Lambda, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from base.common import Constants, Logger, ParseOption, ExitCode
from base.util import Util
from base.data_generator_rnnt import AudioGeneratorForRNNT


class KerasRNNT(object):
  @staticmethod
  def get_rnn(input_vector, vocab_size, num_hidden, num_layers, dropout,
              is_bidirectional=False, output_layer_name=None):
    prev_layer = None

    for _ in range(num_layers):
      lstm_layer = CuDNNLSTM(num_hidden, return_sequences=True)
      prev_layer = prev_layer if prev_layer is not None else input_vector
      if is_bidirectional:
        prev_layer = Bidirectional(lstm_layer, merge_mode='concat')(prev_layer)
      else:
        prev_layer = lstm_layer(prev_layer)

      if dropout > 0:
        prev_layer = Dropout(dropout)(prev_layer)

    if output_layer_name is None:
      output = TimeDistributed(Dense(vocab_size + 1))(prev_layer)
    else:
      output = TimeDistributed(Dense(vocab_size + 1),
                               name=output_layer_name)(prev_layer)
    return Model(inputs=[input_vector], outputs=output)

  @staticmethod
  def create_model(config, vocab, gpus=1):
    # Inputs
    input_tran = Input(name=Constants.INPUT_TRANS, shape=(None, config.feature_dimension))
    input_pred = Input(name=Constants.INPUT_PREDS, shape=(None, len(vocab)))
    input_length = Input(name=Constants.INPUT_INLEN, shape=[1], dtype='int32')
    label_length = Input(name=Constants.INPUT_LBLEN, shape=[1], dtype='int32')
    labels = Input(name=Constants.INPUT_LABEL, shape=[None], dtype='int32')

    # CTC model: Bidirectional LSTM
    encoder = KerasRNNT.get_rnn(input_tran,
                                len(vocab),
                                config.encoder_layer_size,
                                config.encoder_number_of_layer,
                                config.encoder_dropout,
                                config.encoder_rnn_direction == 'bi',
                                Constants.OUTPUT_TRANS)

    # RNN Language Model: Unidirectional LSTM
    decoder = KerasRNNT.get_rnn(input_pred,
                                len(vocab),
                                config.decoder_layer_size,
                                config.decoder_number_of_layer,
                                config.decoder_dropout,
                                False,
                                Constants.OUTPUT_PREDS)

    # declaring lambda function
    loss_out = Lambda(Util.rnnt_lambda_func, output_shape=(1,),
                      name=Constants.LOSS_RNNT)\
      ([encoder.output, decoder.output, labels, input_length, label_length])

    # creating a model
    model = Model(inputs=[encoder.input, decoder.input, labels, input_length,
                          label_length],
                  outputs=loss_out, name="rnnt")

    # multi gpu
    if gpus >= 2:
      from keras.utils.training_utils import multi_gpu_model
      model = multi_gpu_model(model, gpus=gpus)

    # compiling a model
    optimizer = SGD(lr=config.train_learning_rate,
                    decay=config.train_decay,
                    momentum=config.train_momentum,
                    nesterov=config.train_is_nesterov,
                    clipnorm=config.train_clipping_norm)

    model.compile(loss={Constants.LOSS_RNNT: lambda y_true, y_pred: y_pred},
                  optimizer=optimizer)

    return model

def main(): # pylint: disable=too-many-locals
  logger = Logger(name="KerasRNNT", level=Logger.DEBUG).logger

  # Configurations
  config = ParseOption(sys.argv, logger).args
  # Initializing environment variables
  model_name = config.config.split("/")[-1].replace(".conf", "")

  # paths
  checkpoint_dir = config.paths_data_path + "/checkpoints/"
  model_json = config.paths_data_path + "/checkpoints/%s_training.json" % \
            model_name
  model_h5 = config.paths_data_path + "/checkpoints/%s_training.h5" % model_name
  model_pkl = config.paths_data_path + "/checkpoints/%s_ce_loss.pkl"%model_name

  # Cleaning up previous files, if clean_up option was set to True.
  if os.path.isdir(checkpoint_dir):
    if config.paths_clean_up:
      shutil.rmtree(checkpoint_dir)
      logger.info("The previous checkpoints in %s were removed.",
                  checkpoint_dir)
      os.mkdir(checkpoint_dir)
  else:
    os.mkdir(checkpoint_dir)

  # Loading vocabs
  vocab_path = Util.get_file_path(config.paths_data_path, config.paths_vocab)

  if not os.path.isfile(vocab_path):
    logger.critical("%s does not exist.", vocab_path)
    sys.exit(ExitCode.INVALID_FILE_PATH)

  vocab, _ = Util.load_vocab(vocab_path, config=config)
  logger.info("%d words were loaded", len(vocab))
  logger.info("The expanded vocab size : %d", len(vocab) + 1)
  logger.info("The index of a blank symbol: %d", len(vocab))

  model = KerasRNNT.create_model(config=config, vocab=vocab,
                                 gpus=config.device_number_of_gpu)
  model.summary()

  with open(model_json, "w") as json_file:
    json_file.write(model.to_json())
    logger.info("Saved a meta file of a training model to %s", model_json)

  # create a class instance for obtaining batches of data
  audio_gen = AudioGeneratorForRNNT(logger, config, vocab)

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

  # call back function for leaving check points.
  checkpoint = ModelCheckpoint('%s/%s-{epoch:03d}-{loss:03f}-{'
                               'val_loss:03f}.h5'
                               %(checkpoint_dir, model_name),
                               verbose=1, monitor=Constants.VAL_LOSS,
                               save_best_only=False, mode='auto')

  # fitting a model
  hist = model.fit_generator(generator=audio_gen.next_train(),
                             steps_per_epoch=train_batch_size,
                             epochs=config.train_max_epoch,
                             validation_data=audio_gen.next_valid(),
                             validation_steps=valid_batch_size,
                             callbacks=[checkpoint],
                             verbose=1)

  # saving model weights
  model.save_weights(model_h5)
  logger.info("Saved weights of the training model to %s", model_h5)

  # saving model for decoding



  # saving pickle_path
  with open(model_pkl, 'wb') as pkl_file:
    pickle.dump(hist.history, pkl_file)

if __name__ == "__main__":
  main()
