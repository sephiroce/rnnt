# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals, no-member, import-error, no-name-in-module

import sys
import os

from keras.models import model_from_json

from base.util import Util
from base.common import Logger, ParseOption, ExitCode
from base.data_generator_ctc import AudioGeneratorForCTC
from ctc import KerasCTC

def main():
  logger = Logger(name="KmRNNT_CTC_Decoder", level=Logger.DEBUG).logger

  # Configurations
  config = ParseOption(sys.argv, logger).args

  # Loading vocabs
  vocab_path = Util.get_file_path(config.paths_data_path, config.paths_vocab)
  if not os.path.isfile(vocab_path):
    logger.critical("%s does not exist.", vocab_path)
    sys.exit(ExitCode.INVALID_FILE_PATH)
  vocab, id_to_word = Util.load_vocab(vocab_path, config=config)
  logger.info("%d words were loaded.", len(vocab))

  json_file = open(Util.get_file_path(config.paths_data_path,
                                      config.paths_model_json), "r")
  loaded_model_json = json_file.read()
  json_file.close()

  model = model_from_json(loaded_model_json)
  model_weight_path = Util.get_file_path(config.paths_data_path,
                                         config.paths_model_h5)
  model.load_weights(model_weight_path)
  model.summary()

  audio_gen = AudioGeneratorForCTC(logger, config, vocab)

  # CMVN
  audio_gen.load_train_data(Util.get_file_path(config.paths_data_path,
                                               "train_corpus.json"), 1000)
  # Testing data
  audio_gen.load_test_data(Util.get_file_path(config.paths_data_path,
                                              "test_corpus.json"))

  with open(model_weight_path+".utt", "w") as utt_file:
    for i, val in enumerate(audio_gen.next_test()):
      if i == len(audio_gen.test_audio_paths):
        break
      result = KerasCTC.get_result_str(model.predict(val[0]), id_to_word)
      logger.info("UTT%03d: %s", i+1, result)
      utt_file.write("%s (spk-%d)\n"%(result, i+1))
  logger.info("UTT File saved into %s.utt", sys.argv[4])

if __name__ == "__main__":
  main()
