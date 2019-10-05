# -*- coding: utf-8 -*-
# pylint: disable=too-many-instance-attributes, too-many-arguments
# pylint: disable=too-many-locals, too-many-branches, too-many-statements

"""
This is a customized version of an AudioGenerator class of lucko515.
https://github.com/lucko515/speech-recognition-neural-network/blob/master/data_generator.py

Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""
import json
import random
import sys
import numpy as np

import scipy.io.wavfile as wav
from python_speech_features import mfcc
from base.common import Constants
from base.utils import KmRNNTUtil as Util

RNG_SEED = 123

class AudioGeneratorForCTC:
  def __init__(self, logger, config, vocab):
    """
    Params:
      step (int): Step size in milliseconds between windows (for spectrogram 
      ONLY)
      desc_file (str, optional): Path to a JSON-line file that contains
        labels and paths to the audio files. If this is None, then
        load metadata right away

    """

    self.logger = logger
    self.basepath = config.paths_data_path

    self.max_freq = 16000
    self.feat_dim = config.feature_dimension
    self.feats_mean = None
    self.feats_std = None
    self.rng = random.Random(RNG_SEED)
    self.cur_train_index = 0
    self.cur_valid_index = 0
    self.cur_test_index = 0
    self.max_duration = config.prep_max_duration
    self.minibatch_size = config.train_batch
    self.vocab = vocab
    self.is_char = config.prep_text_unit == Constants.CHAR
    self.feat_type = config.feature_type

    self.train_audio_paths = Constants.EMPTY
    self.train_durations = 0
    self.train_texts = Constants.EMPTY

    self.valid_audio_paths = Constants.EMPTY
    self.valid_durations = 0
    self.valid_texts = Constants.EMPTY

    self.test_audio_paths = Constants.EMPTY
    self.test_durations = 0
    self.test_texts = Constants.EMPTY

  def get_batch(self, partition):
    """ Obtain a batch of train, validation, or test data
    """
    if partition == Constants.TRAINING:
      audio_paths = self.train_audio_paths
      cur_index = self.cur_train_index
      texts = self.train_texts
    elif partition == Constants.VALIDATION:
      audio_paths = self.valid_audio_paths
      cur_index = self.cur_valid_index
      texts = self.valid_texts
    elif partition == Constants.EVALUATION:
      audio_paths = self.test_audio_paths
      cur_index = self.cur_test_index
      texts = self.test_texts
    else:
      raise Exception("Invalid partition. Must be %s/%s"%
                      (Constants.TRAINING, Constants.VALIDATION))

    # extracting features
    features = [self.normalize(self.featurize(a)) for a in
                audio_paths[cur_index:cur_index+self.minibatch_size]]

    # calculate necessary sizes
    batch_size = min(len(features), self.minibatch_size)
    max_length = max([features[i].shape[0] for i in range(0, batch_size)])

    # plus two for BOS and EOS
    if self.is_char:
      max_string_length = max([len(texts[cur_index + i])
                               for i in range(0, batch_size)])
    else:
      max_string_length = max([len(texts[cur_index+i].split(" "))
                               for i in range(0, batch_size)])

    # initialize the arrays
    x_data = np.zeros([batch_size, max_length, self.feat_dim])
    labels = np.ones([batch_size, max_string_length]) * len(self.vocab)

    input_length = np.zeros([batch_size, 1])
    label_length = np.zeros([batch_size, 1])

    for i in range(0, batch_size):
      # assigns feature inputs and their lengths
      feat = features[i]
      input_length[i] = feat.shape[0] # T
      x_data[i, :feat.shape[0], :] = feat # x

      # calculate labels & label_length
      int_seq = list()

      if self.is_char:
        for char in texts[cur_index + i].strip():
          if char in self.vocab:
            int_seq.append(self.vocab[char])
          elif char == ' ':
            int_seq.append(self.vocab[Constants.SPACE])
          else:
            self.logger.error(texts[cur_index + i].strip())
            int_seq.append(self.vocab[Constants.UNK])
      else:
        for bpe in texts[cur_index+i].strip().split(" "):
          if bpe in self.vocab:
            int_seq.append(self.vocab[bpe])
          else:
            int_seq.append(self.vocab[Constants.UNK])

      # int_seq = y, [0] + int_seq -> prepanded
      label = np.array(int_seq)
      label_length[i] = len(label)
      labels[i, :len(label)] = label

    # return the arrays
    inputs = {Constants.KEY_INPUT: x_data,
              Constants.KEY_LABEL: labels,
              Constants.KEY_INLEN: input_length,
              Constants.KEY_LBLEN: label_length
              }

    outputs = {Constants.KEY_CTCLS: np.zeros([batch_size])}

    return inputs, outputs

  def shuffle_data_by_partition(self, partition):
    """ Shuffle the training or validation data
    """
    if partition == Constants.TRAINING:
      self.train_audio_paths, self.train_durations, self.train_texts = shuffle_data(
          self.train_audio_paths, self.train_durations, self.train_texts)
    elif partition == Constants.VALIDATION:
      self.valid_audio_paths, self.valid_durations, self.valid_texts = shuffle_data(
          self.valid_audio_paths, self.valid_durations, self.valid_texts)
    else:
      raise Exception("Invalid partition. Must be train/validation")

  def sort_data_by_duration(self, partition):
    """ Sort the training or validation sets by (increasing) duration
    """
    if partition == Constants.TRAINING:
      self.train_audio_paths, self.train_durations, self.train_texts = \
          sort_data(self.train_audio_paths, self.train_durations,
                    self.train_texts)
    elif partition == Constants.VALIDATION:
      self.valid_audio_paths, self.valid_durations, self.valid_texts = \
          sort_data(self.valid_audio_paths, self.valid_durations,
                    self.valid_texts)
    else:
      raise Exception("Invalid partition. Must be %s/%s" %
                      (Constants.TRAINING, Constants.VALIDATION))

  def next_train(self):
    """ Obtain a batch of training data
    """
    while True:
      ret = self.get_batch(Constants.TRAINING)
      self.cur_train_index += self.minibatch_size
      if self.cur_train_index > len(self.train_texts) - self.minibatch_size:
        self.cur_train_index = 0
        self.shuffle_data_by_partition(Constants.TRAINING)
      yield ret

  def next_valid(self):
    """ Obtain a batch of validation data
    """
    while True:
      ret = self.get_batch(Constants.VALIDATION)
      self.cur_valid_index += self.minibatch_size
      if self.cur_valid_index > len(self.valid_texts) - self.minibatch_size:
        self.cur_valid_index = 0
        self.shuffle_data_by_partition(Constants.VALIDATION)
      yield ret

  def next_test(self):
    """ Obtain a batch of test data
    """
    while True:
      ret = self.get_batch(Constants.EVALUATION)
      self.cur_test_index += self.minibatch_size
      if self.cur_test_index > len(self.test_texts) - self.minibatch_size:
        self.cur_test_index = 0
      yield ret

  def load_train_data(self, desc_file='train_corpus.json', cmvn_samples=100):
    self.load_metadata_from_desc_file(desc_file, Constants.TRAINING)
    self.fit_train(cmvn_samples)

  def load_validation_data(self, desc_file='valid_corpus.json'):
    self.load_metadata_from_desc_file(desc_file, Constants.VALIDATION)

  def load_test_data(self, desc_file='test_corpus.json'):
    self.load_metadata_from_desc_file(desc_file, Constants.EVALUATION)

  def load_metadata_from_desc_file(self, desc_file, partition):
    """ Read metadata from a JSON-line file
      (possibly takes long, depending on the filesize)
    Params:
      desc_file (str):  Path to a JSON-line file that contains labels and
        paths to the audio files
      partition (str): One of 'train', 'validation' or 'test'
    """
    audio_paths, durations, texts = [], [], []
    with open(desc_file) as json_line_file:
      for line_num, json_line in enumerate(json_line_file):
        try:
          spec = json.loads(json_line)
          if 0 < self.max_duration < float(spec[Constants.DURATION]):
            continue
          audio_paths.append(spec[Constants.KEY])
          durations.append(float(spec[Constants.DURATION]))
          texts.append(spec[Constants.TEXT])
        except json.decoder.JSONDecodeError as err:
          # json module version
          self.logger.error('Error reading line #{}: {}, {}'.format(line_num,
                                                                    json_line,
                                                                    err.msg))
    if partition == Constants.TRAINING:
      self.train_audio_paths = audio_paths
      self.train_durations = durations
      self.train_texts = texts
    elif partition == Constants.VALIDATION:
      self.valid_audio_paths = audio_paths
      self.valid_durations = durations
      self.valid_texts = texts
    elif partition == Constants.EVALUATION:
      self.test_audio_paths = audio_paths
      self.test_durations = durations
      self.test_texts = texts
    else:
      raise Exception("Invalid partition to load metadata. Must be %s/%s/%s"
                      % (Constants.TRAINING, Constants.VALIDATION,
                         Constants.EVALUATION))

  def fit_train(self, k_samples=100):
    """ Estimate the mean and std of the features from the training set
    Params:
      k_samples (int): Use this number of samples for estimation
    """
    # calculating CMVN if it was not pre-calculated.
    if self.feats_mean is None or self.feats_std is None:
      k_samples = min(k_samples, len(self.train_audio_paths))
      samples = self.rng.sample(self.train_audio_paths, k_samples)
      feats = [self.featurize(s) for s in samples]
      feats = np.vstack(feats)
      self.feats_mean = np.mean(feats, axis=0)
      self.feats_std = np.std(feats, axis=0)

  def featurize(self, audio_clip):
    """ For a given audio clip, calculate the corresponding feature
    Params:
      audio_clip (str): Path to the audio clip
    """
    if self.feat_type == Constants.FEAT_MFCC:
      (rate, sig) = wav.read(self.basepath + "/" + audio_clip)
      return mfcc(sig, rate, numcep=self.feat_dim)

    if self.feat_type == Constants.FEAT_FBANK:
      return Util.get_fbanks(self.basepath + "/" + audio_clip,
                             frame_size=0.025, frame_stride=0.01,
                             n_filt=self.feat_dim)

    self.logger.error("%s is not supported yet.", self.feat_type)
    sys.exit(1)

  def normalize(self, feature, eps=1e-14):
    """ Center a feature using the mean and std
    Params:
      feature (numpy.ndarray): Feature to normalize
    """
    return (feature - self.feats_mean + eps) / (self.feats_std + eps)

def shuffle_data(audio_paths, durations, texts):
  """ Shuffle the data (called after making a complete pass through
    training or validation data during the training process)
  Params:
    audio_paths (list): Paths to audio clips
    durations (list): Durations of utterances for each audio clip
    texts (list): Sentences uttered in each audio clip
  """
  perm = np.random.permutation(len(audio_paths))
  audio_paths = [audio_paths[i] for i in perm]
  durations = [durations[i] for i in perm]
  texts = [texts[i] for i in perm]
  return audio_paths, durations, texts

def sort_data(audio_paths, durations, texts):
  """ Sort the data by duration
  Params:
    audio_paths (list): Paths to audio clips
    durations (list): Durations of utterances for each audio clip
    texts (list): Sentences uttered in each audio clip
  """
  sort = np.argsort(durations).tolist()
  audio_paths = [audio_paths[i] for i in sort]
  durations = [durations[i] for i in sort]
  texts = [texts[i] for i in sort]
  return audio_paths, durations, texts
