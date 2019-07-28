# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods

import logging

class Constants:
  TRAINING = "Train"
  VALIDATION = "Valid"
  EVALUATION = "Test"
  EMPTY = "__empty__"
  EPS = '<eps>'
  BLANK = 'blank'
  SPACE = 'space'
  UNK = 'u'
  BOS = '<s>'
  EOS = '</s>'
  DURATION = 'duration'
  KEY = 'key'
  TEXT = 'text'
  LOG_FBANK = 'log_filterbank'

class ExitCode:
  NO_DATA = 0
  NOT_SUPPORTED = 1
  INVALID_OPTION = 11
  INVALID_CONVERSION = 12
  INVALID_NAME = 13
  INVALID_NAME_OF_CONFIGURATION_FILE = 14
  INVALID_FILE_PATH = 15
  INVALID_DICTIONARY = 16

class Logger:
  """
  !!Usage: please create a logger with one line as shown in the example below.
    logger = Logger(name = "word2vec", level = Logger.DEBUG).logger
    logger.debug('debug message')
    logger.info('info message')
    logger.warn('warn message')
    logger.error('error message')
    logger.critical('critical message')

  This logger print out logging messages similar to the logging message of tensorflow.
  2018-07-01 19:35:33.945120: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406]
  2018-07-20 16:23:08.000295: I kmlm_common.py:94] Configuration lists:

  TO-DO:
  verbose level
  """
  DEBUG = logging.DEBUG
  NOTSET = logging.NOTSET
  INFO = logging.INFO
  WARN = logging.WARN
  ERROR = logging.ERROR
  CRITICAL = logging.CRITICAL

  def __init__(self, name="__default__", level=logging.NOTSET):
    self.logger = logging.getLogger(name)
    self.logger.setLevel(level)
    handle = logging.StreamHandler()
    handle.setLevel(level)
    formatter = logging.Formatter('%(asctime)s: %(levelname).1s '
                                  '%(filename)s:%(lineno)d] %(message)s')
    formatter.default_msec_format = '%s.%06d'
    handle.setFormatter(formatter)
    self.logger.addHandler(handle)
