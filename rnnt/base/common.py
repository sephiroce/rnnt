# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods, too-many-locals, no-member,
# pylint: disable=too-many-statements

"""common.py: global functionaries for python programs"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import logging
import os
import sys
import argparse

class Constants(object): # pylint: disable=no-init
  TRAINING = "Train"
  VALIDATION = "Valid"
  EVALUATION = "Test"
  EMPTY = "__empty__"
  EPS = '<eps>'
  SPACE = '<SPACE>'
  UNK = 'u'
  BOS = '<s>'
  EOS = '</s>'
  DURATION = 'duration'
  KEY = 'key'
  TEXT = 'text'
  LOG_FBANK = 'log_filterbank'
  VAL_LOSS = 'val_loss'
  WORD = 'word'
  CHAR = 'char'
  BI_DIRECTION = 'bi'

  KEY_CTCDE = 'KEY_CTCDE'

  INPUT_TRANS = 'INPUT_TRANS'
  INPUT_PREDS = 'INPUT_PREDS'
  INPUT_LABEL = 'INPUT_LABEL'
  INPUT_INLEN = 'INPUT_INLEN'
  INPUT_LBLEN = 'INPUT_LBLEN'
  OUTPUT_TRANS = 'OUTPUT_TRANS'
  OUTPUT_PREDS = 'OUTPUT_PREDS'

  LOSS_CTC = 'LOSS_CTC'
  LOSS_RNNT = 'LOSS_RNNT'

  FEAT_MFCC = 'mfcc'
  FEAT_FBANK = 'fbank'

class CmvnFiles(object): # pylint: disable=no-init
  mean = "cmvn.mean"
  std = "cmvn.std"

class ExitCode(object): # pylint: disable=no-init
  NO_DATA = 0
  NOT_SUPPORTED = 1
  INVALID_OPTION = 11
  INVALID_CONVERSION = 12
  INVALID_NAME = 13
  INVALID_NAME_OF_CONFIGURATION_FILE = 14
  INVALID_FILE_PATH = 15
  INVALID_DICTIONARY = 16

class Logger(object):
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
    self.logger.propagate = False
    self.logger.addHandler(handle)

class ParseOption(object):
  """
  it merges options from both an option file and command line into python option
  """
  def __init__(self, argv, logger):
    self.logger = logger

    parser = self.build_parser()
    dparser = self.build_parser()

    if len(argv) > 1:
      # args from command line
      command_args = parser.parse_args(argv[1:])
      default_args = dparser.parse_args([])
      # args from config file
      if not command_args.config.endswith(".conf"):
        self.logger.critical("The the extension of configuration file must be "
                             "conf, but %s" %command_args.config)
        sys.exit(ExitCode.INVALID_NAME_OF_CONFIGURATION_FILE)

      data_path = command_args.paths_data_path
      file_path = command_args.config

      if data_path and not os.path.exists(file_path):
        file_path = data_path + "/" + file_path

      config_args = parser.parse_args(["@" + file_path])

      # merge command args and file args into args
      command_dict = vars(command_args)
      if "config" not in command_dict:
        self.logger.critical("\"config\" is a required option for the command "
                             "line.")
        sys.exit(ExitCode.INVALID_OPTION)
      config_dict = vars(config_args)
      default_dict = vars(default_args)

      for arg in default_dict:
        if arg != "config" and arg not in command_dict or \
          command_dict[arg] is None or \
          command_dict[arg] == default_dict[arg]:
          command_dict[arg] = config_dict[arg]

      # adjusting options to be consistent
      if command_dict["prep_use_beos"]:
        command_dict["prep_use_bos"] = False
        command_dict["prep_use_eos"] = False

      # I want to save argument as a namespace..
      args = argparse.Namespace(**command_dict)

      # Checking sanity of the configurations
      if not self.sanity_check(args):
        sys.exit(ExitCode.INVALID_OPTION)

      # Modifying hyper-parameters depends on the number of GPUs
      if args.device_number_of_gpu > 1:
        tmp_lr = args.train_learning_rate
        tmp_bt = args.train_batch
        args.train_learning_rate /= args.device_number_of_gpu
        args.train_batch *= args.device_number_of_gpu
        self.logger.info("According to the number of GPUs: %d,",
                         args.device_number_of_gpu)
        self.logger.info("train_learning_rate was modified from %.3f to "
                         "%.3f.", tmp_lr, args.train_learning_rate)
        self.logger.info("train_batch was modified from %d to "
                         "%d.", tmp_bt, args.train_batch)

      self.print_args(args)

      self._args = args
    else:
      self.logger.critical("No options..")
      sys.exit(ExitCode.INVALID_OPTION)

  @staticmethod
  def str2bool(bool_string):
    return bool_string.lower() in ("yes", "true", "t", "1")

  @property
  def args(self):
    return self._args

  def sanity_check(self, args):
    # Checking sanity of configuration options
    if not args.paths_data_path:
      self.logger.critical("the following arguments are required: --paths-data-path")
      return False

    if not os.path.isdir(args.paths_data_path)\
        or os.path.isfile(args.paths_data_path):
      self.logger.critical("A data path must exist, please check the data path "
                           "option : %s" % args.paths_data_path)
      return False

    return True

  def print_args(self, args):
    self.logger.info("/******************************************")
    self.logger.info("        Settings")
    self.logger.info("*******************************************")
    sorted_args = sorted(vars(args))
    pre_name = ""
    for arg in sorted_args:
      name = arg.split("_")[0]
      if name != pre_name:
        self.logger.info(". %s"%name.upper())
        pre_name = name

      self.logger.info("- %s=%s"%(arg, getattr(args, arg)))
    self.logger.info("*******************************************/")

  @staticmethod
  def build_parser():
    # create parser
    parser = argparse.ArgumentParser(description="Keras based RNN-LM Toolkit ",
                                     fromfile_prefix_chars='@')
    parser.add_argument("--config",
                        help="options can be loaded from this config file")

    # Pre-processing
    prep_group = parser.add_argument_group(title="pre-processing",
                                           description="options related to "
                                                       "text pre-processing")
    prep_group.add_argument('--prep-use-unk', default=False, type=ParseOption.str2bool,
                            help="a unk (unknown symbol) is used or not.")
    prep_group.add_argument('--prep-use-bos', default=False, type=ParseOption.str2bool,
                            help="Whether a bos (beginning of sentence) "
                                 "symbol is used or not.")
    prep_group.add_argument('--prep-use-eos', default=False, type=ParseOption.str2bool,
                            help="Whether a eos (end of sentence) symbol is "
                                 "used or not.")
    prep_group.add_argument('--prep-use-beos', default=False, type=ParseOption.str2bool,
                            help="Whether symbols for bos and eos are used "
                                 "as one symbol.")
    prep_group.add_argument('--prep-max-string-length', type=int, default=-1,
                            help="Maximum number of tokens in a sentence, "
                                 "-1 means infinite")
    prep_group.add_argument('--prep-text-unit', default=Constants.WORD,
                            help="Only %s and %s can be accepted"%(
                                Constants.WORD, Constants.CHAR))
    prep_group.add_argument("--prep-max-duration", type=int, default=-1,
                            help="max duration of input speech in seconds")
    prep_group.add_argument("--prep-cmvn-samples", type=int, default=-1,
                            help="the number of samples for cmvn")

    # Hyper-parameters for training
    train_group = \
      parser.add_argument_group(title="training",
                                description="Hyper-parameters for model "
                                            "architecture (some of them "
                                            "might be specialized for "
                                            "specific model architecture)")
    train_group.add_argument('--train-max-epoch', type=int, default=1,
                             help="maximum epoch")
    train_group.add_argument('--train-batch', type=int, default=-1,
                             help="the size of batch data which is the unit "
                                  "for calculating gradient.")
    train_group.add_argument('--train-learning-rate', type=float,
                             help="learning rate for optimizer")
    train_group.add_argument('--train-optimizer', help="type of optimizer")
    train_group.add_argument('--train-earlystop-patience',
                             help="the number of epoch for earlystop patience",
                             type=int, default=1000000)
    train_group.add_argument('--train-plateau-patience',
                             help="the number of epoch for plateau patience",
                             type=int, default=1000000)
    train_group.add_argument('--train-earlystop-delta',
                             help="the min delta counting the epoch numbers "
                                  "for earlystop patience", type=float,
                             default=1e-4)
    train_group.add_argument('--train-plateau-delta',
                             help="the min delta counting the epoch numbers "
                                  "for plateau patience", type=float,
                             default=1e-4)
    train_group.add_argument('--train-plateau-factor',
                             help="the factor for plateau", type=float,
                             default=0.1)
    train_group.add_argument('--train-norm-l2', type=float, default=0.0,
                             help="Scaling factor for L2 norm, if it is 0.0 "
                                  "then L2 norm won't be calculated.")
    train_group.add_argument("--train-clipping-norm", type=float,
                             default=5.0,
                             help="norm for clipping gradients")
    train_group.add_argument("--train-decay", type=float,
                             default=0.0, help="decaying learning rate")
    train_group.add_argument("--train-momentum", type=float,
                             default=0.9, help="momentum for learning rate")
    train_group.add_argument("--train-is-nesterov", type=ParseOption.str2bool,
                             default="True", help="is using nesterov momentum?")
    train_group.add_argument("--train-gaussian-noise", type=float,
                             default=0.0, help="gaussian weight noise")

    # Paths
    path_group = parser.add_argument_group(title="paths",
                                           description="paths for inout and "
                                                       "output files")
    path_group.add_argument("--paths-model", default=None,
                            help="model_path.json, model_path.h5")
    path_group.add_argument("--paths-data-path", help="base path")
    path_group.add_argument("--paths-vocab",
                            help="vocab file")
    path_group.add_argument('--paths-clean-up', type=ParseOption.str2bool, default=False,
                            help="cleaning up previous models and graphs")
    path_group.add_argument('--paths-model-json', default=None,
                            help="model json")
    path_group.add_argument('--paths-model-h5', default=None,
                            help="model h5")
    path_group.add_argument('--paths-train-corpus', default="train_corpus.json",
                            help="train corpus")
    path_group.add_argument('--paths-valid-corpus', default="valid_corpus.json",
                            help="valid corpus")
    path_group.add_argument('--paths-test-corpus', default="test_corpus.json",
                            help="test corpus")

    # Feature
    feature_group = parser.add_argument_group(title="feature",
                                              description="speech feature")
    feature_group.add_argument("--feature-type", default="mfcc",
                               help="mfcc or fbank")
    feature_group.add_argument("--feature-dimension", type=int, default=40,
                               help="feature dimension")

    # Encoder architecture
    encoder_group = parser.add_argument_group(title="encoder architecture",
                                              description="hyper-parameter "
                                                          "for each layers")
    encoder_group.add_argument("--encoder-layer-size", type=int,
                               help="size of a hidden layer")
    encoder_group.add_argument("--encoder-number-of-layer", type=int,
                               help="number of hidden layers")
    encoder_group.add_argument("--encoder-rnn-direction", default="bi",
                               help="uni or bi")
    encoder_group.add_argument("--encoder-dropout", type=float, default=0.0,
                               help="drop out, default = 0 to keep all")

    # Decoder architecture
    decoder_group = parser.add_argument_group(title="decoder architecture",
                                              description="hyper-parameter for "
                                                          "each layers")
    decoder_group.add_argument("--decoder-layer-size", type=int,
                               help="size of a hidden layer")
    decoder_group.add_argument("--decoder-number-of-layer", type=int,
                               help="number of hidden layers")
    encoder_group.add_argument("--decoder-dropout", type=float, default=0.0,
                               help="drop out, default = 0 to keep all")

    # Setting for the entire model
    model_group = parser.add_argument_group(title="model architecture",
                                            description="hyper-parameter for "
                                                        "the entire model")
    model_group.add_argument("--model-init-scale", type=float,
                             default=0.1,
                             help="Initial scale of variables [-val, val], "
                                  "default = 0.1")

    # Inference option
    inference_group = parser.add_argument_group()
    inference_group.add_argument("--inference-is-debug", type=ParseOption.str2bool, default="False",
                                 help="true means storing softmax sequences to check")
    inference_group.add_argument("--inference-beam-width", type=int, default=12,
                                 help="beam width for beam search decoding")

    # Device
    device_group = parser.add_argument_group(title="device")
    device_group.add_argument("--device-number-of-gpu", type=int, default=1)

    return parser

def main():
  logger = Logger(name="RNN-T Configurations", level=Logger.DEBUG).logger

  # Configurations
  ParseOption(sys.argv, logger)

if __name__ == "__main__":
  main()
