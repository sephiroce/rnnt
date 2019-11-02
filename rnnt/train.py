# -*- coding: utf-8 -*-
# pylint: disable=no-member, import-error, no-name-in-module, too-many-arguments

"""train.py: A training script"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import os
import pickle
import glob
import sys
import time

from keras.callbacks import ModelCheckpoint, EarlyStopping

from rnnt.base.common import Constants, Logger, ParseOption
from rnnt.base.util import Util
from rnnt.base.data_generator import AudioGeneratorForRNNT, AudioGeneratorForCTC
from rnnt.keras_model import KerasModel

def main():  # pylint: disable=too-many-locals
  current_milli_time = int(round(time.time() * 1000))
  logger = Logger(name="KerasSeq2SeqASR", level=Logger.DEBUG).logger

  # Get configurations
  config = ParseOption(sys.argv, logger).args

  if not config.decoder_layer_size or not config.decoder_number_of_layer:
    model_type = "CTC"
    model_name = \
      "%s_%d_%s%d_%s%dx%d_lr%.6f_%dgpus" % (model_type,
                                            current_milli_time,
                                            config.feature_type,
                                            config.feature_dimension,
                                            config.encoder_rnn_direction,
                                            config.encoder_number_of_layer,
                                            config.encoder_layer_size,
                                            config.train_learning_rate,
                                            config.device_number_of_gpu)
  else:
    model_type = "RNNT"
    model_name = \
      "%s_%d_%s%d_%s%dx%d.%dx%d_lr%.6f_%dgpus" % (model_type,
                                                  current_milli_time,
                                                  config.feature_type,
                                                  config.feature_dimension,
                                                  config.encoder_rnn_direction,
                                                  config.encoder_number_of_layer,
                                                  config.encoder_layer_size,
                                                  config.decoder_number_of_layer,
                                                  config.decoder_layer_size,
                                                  config.train_learning_rate,
                                                  config.device_number_of_gpu)

  # paths
  checkpoint_dir = config.paths_data_path + "/checkpoints/"
  model_json = config.paths_data_path + \
               "/checkpoints/%s_training.json" % model_name
  model_h5 = config.paths_data_path + \
             "/checkpoints/%s_training.h5" % model_name
  model_pkl = config.paths_data_path + \
              "/checkpoints/%s_ce_loss.pkl" % model_name

  # Cleaning up previous files, if clean_up option was set to True.
  if os.path.isdir(checkpoint_dir):
    if config.paths_clean_up:
      ckpt_files = glob.glob(Util.get_file_path(config.paths_data_path,
                                                "*%s*"%model_name))
      for ckpt_file in ckpt_files:
        os.remove(ckpt_file)
        logger.info("%s was removed.", ckpt_file)
  else:
    os.mkdir(checkpoint_dir)

  # Loading vocabs
  vocab, _ = Util.load_vocab(Util.get_file_path(config.paths_data_path,
                                                config.paths_vocab),
                             config=config)

  # Creating a model
  model = KerasModel.create_model(config=config, vocab=vocab,
                                  is_rnnt=model_type == "RNNT")

  # Logging information
  logger.info("Model name: %s", model_name)
  logger.info("%d words were loaded", len(vocab))
  logger.info("The expanded vocab size : %d", len(vocab) + 1)
  logger.info("The index of a blank symbol: %d", len(vocab))
  model.summary()

  # make checkpoints/ directory, if necessary
  if not os.path.exists(config.paths_data_path + '/checkpoints'):
    os.makedirs(config.paths_data_path + '/checkpoints')

  # saving the meta files of models
  with open(model_json, "w") as json_file:
    json_file.write(model.to_json())
    logger.info("Saved a meta file of a training model to %s", model_json)

  # create a class instance for obtaining batches of data
  if model_type == "RNNT":
    audio_gen = AudioGeneratorForRNNT(logger, config, vocab)
  else:
    audio_gen = AudioGeneratorForCTC(logger, config, vocab)

  # add the training data to the generator
  audio_gen.load_train_data(Util.get_file_path(config.paths_data_path,
                                               config.paths_train_corpus),
                            config.prep_cmvn_samples)
  audio_gen.load_validation_data(Util.get_file_path(config.paths_data_path,
                                                    config.paths_valid_corpus))
  audio_gen.load_test_data(Util.get_file_path(config.paths_data_path,
                                              config.paths_test_corpus))

  # train a new model
  train_batch_size = (len(audio_gen.train_audio_paths) // config.train_batch)
  valid_batch_size = (len(audio_gen.valid_audio_paths) // config.train_batch)

  # call back functions
  early_stopping = EarlyStopping(monitor=Constants.VAL_LOSS, patience=200,
                                 verbose=1, mode='min')

  # call back function for leaving check points.
  checkpoint = ModelCheckpoint('%s/%s-{epoch:03d}-{loss:03f}-{val_loss:03f}.h5'
                               % (checkpoint_dir, model_name), verbose=1,
                               monitor=Constants.VAL_LOSS, save_best_only=False,
                               mode='auto')

  # fitting a model
  hist = model.fit_generator(generator=audio_gen.next_train(),
                             steps_per_epoch=train_batch_size,
                             epochs=config.train_max_epoch,
                             validation_data=audio_gen.next_valid(),
                             validation_steps=valid_batch_size,
                             callbacks=[early_stopping, checkpoint],
                             verbose=1)

  # saving model weights
  model.save_weights(model_h5)
  logger.info("Saved weights of the model to %s", model_h5)

  # saving pickle_path
  with open(model_pkl, 'wb') as pkl_file:
    pickle.dump(hist.history, pkl_file)

if __name__ == "__main__":
  main()
