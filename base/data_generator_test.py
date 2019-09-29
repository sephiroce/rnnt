# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
This is not a completed unit test code.
"""
import os
import sys
from base.data_generator import AudioGenerator
from base.utils import KmRNNTUtil as Util
from base.common import Logger, ParseOption, ExitCode

def main():
  logger = Logger(name="Audio Data generator test", level=Logger.INFO).logger
  sys.argv.append("--config=test.conf")
  sys.argv.append("--paths-data-path=base/test")
  config = ParseOption(sys.argv, logger).args

  # Loading a vocabulary
  vocab_path = Util.get_file_path(config.paths_data_path, config.paths_vocab)
  if not os.path.isfile(vocab_path):
    logger.critical("%s does not exist.", vocab_path)
    sys.exit(ExitCode.INVALID_FILE_PATH)
  vocab, _ = Util.load_vocab(vocab_path, config=config)

  audio_gen = AudioGenerator(logger, config, vocab)

  #add the training data to the generator
  audio_gen.load_train_data("base/test/test_corpus.json")

  for i, value in enumerate(audio_gen.next_train()):
    if i >= 2:
      break
    logger.info(value)

if __name__ == "__main__":
  main()
