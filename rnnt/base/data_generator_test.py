# -*- coding: utf-8 -*-
# pylint: disable=no-member, no-name-in-module, import-error

"""data_generator_test.py: To see data are generated as intended."""

import os
import sys
from rnnt.base.data_generator import AudioGeneratorForCTC, AudioGeneratorForRNNT
from rnnt.base.util import Util
from rnnt.base.common import Logger, ParseOption, ExitCode

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"


def main():
  logger = Logger(name="Audio Data generator test", level=Logger.INFO).logger
  sys.argv.append("--config=test.conf")
  sys.argv.append("--paths-data-path=rnnt/base/test")
  config = ParseOption(sys.argv, logger).args

  # Loading a vocabulary
  vocab_path = Util.get_file_path(config.paths_data_path, config.paths_vocab)
  if not os.path.isfile(vocab_path):
    logger.critical("%s does not exist.", vocab_path)
    sys.exit(ExitCode.INVALID_FILE_PATH)
  vocab, _ = Util.load_vocab(vocab_path, config=config)

  audio_gen = AudioGeneratorForCTC(logger, config, vocab)
  audio_gen.load_train_data("rnnt/base/test/test_corpus.json")

  for i, value in enumerate(audio_gen.next_train()):
    if i >= 2:
      break
    logger.info(value)

  audio_gen = AudioGeneratorForRNNT(logger, config, vocab)
  audio_gen.load_train_data("rnnt/base/test/test_corpus.json")

  for i, value in enumerate(audio_gen.next_train()):
    if i >= 2:
      break
    logger.info(value)

if __name__ == "__main__":
  main()
