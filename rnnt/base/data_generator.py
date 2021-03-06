# -*- coding: utf-8 -*-
# pylint: disable=import-error, too-many-instance-attributes, useless-super-delegation
# pylint: disable=too-many-locals, too-many-branches

"""data_generator.py: memory based data generator both for CTC and for RNN-T

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
from python_speech_features import mfcc, fbank
from python_speech_features.base import delta
from rnnt.base.common import Constants, ProcessType
from rnnt.base.util import Util

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

RNG_SEED = 123

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

class AudioGenerator(object):
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
    self.feats_mean = \
      np.loadtxt(Util.get_file_path(self.basepath, config.paths_cmvn_mean)) \
      if config.paths_cmvn_mean else None
    self.feats_std = \
      np.loadtxt(Util.get_file_path(self.basepath, config.paths_cmvn_std)) \
      if config.paths_cmvn_std else None
    self.rng = random.Random(RNG_SEED)
    self.cur_train_index = 0
    self.cur_valid_index = 0
    self.cur_test_index = 0
    self.max_duration = config.prep_max_duration
    self.minibatch_size = config.train_batch
    self.vocab = vocab
    self.is_char = config.prep_text_unit == Constants.CHAR
    self.feat_type = config.feature_type

    self.train_audio_paths = None
    self.train_durations = 0
    self.train_texts = None

    self.valid_audio_paths = None
    self.valid_durations = 0
    self.valid_texts = None

    self.test_audio_paths = None
    self.test_durations = 0
    self.test_texts = None

  def shuffle_data_by_partition(self, partition):
    """ Shuffle the training or validation data
    """
    if partition == ProcessType.TRAINING:
      self.train_audio_paths, self.train_durations, self.train_texts = shuffle_data(
          self.train_audio_paths, self.train_durations, self.train_texts)
    elif partition == ProcessType.VALIDATION:
      self.valid_audio_paths, self.valid_durations, self.valid_texts = shuffle_data(
          self.valid_audio_paths, self.valid_durations, self.valid_texts)
    else:
      raise Exception("Invalid partition. Must be train/validation")

  def sort_data_by_duration(self, partition):
    """ Sort the training or validation sets by (increasing) duration
    """
    if partition == ProcessType.TRAINING:
      self.train_audio_paths, self.train_durations, self.train_texts = \
          sort_data(self.train_audio_paths, self.train_durations,
                    self.train_texts)
    elif partition == ProcessType.VALIDATION:
      self.valid_audio_paths, self.valid_durations, self.valid_texts = \
          sort_data(self.valid_audio_paths, self.valid_durations,
                    self.valid_texts)
    else:
      raise Exception("Invalid partition. Must be %s/%s" %
                      (ProcessType.TRAINING, ProcessType.VALIDATION))

  def get_batch(self, partition):
    raise NotImplementedError()

  def next_train(self):
    """ Obtain a batch of training data
    """
    while True:
      ret = self.get_batch(ProcessType.TRAINING)
      self.cur_train_index += self.minibatch_size
      if self.cur_train_index > len(self.train_texts) - self.minibatch_size:
        self.cur_train_index = 0
        self.shuffle_data_by_partition(ProcessType.TRAINING)
      yield ret

  def next_valid(self):
    """ Obtain a batch of validation data
    """
    while True:
      ret = self.get_batch(ProcessType.VALIDATION)
      self.cur_valid_index += self.minibatch_size
      if self.cur_valid_index > len(self.valid_texts) - self.minibatch_size:
        self.cur_valid_index = 0
        self.shuffle_data_by_partition(ProcessType.VALIDATION)
      yield ret

  def next_test(self):
    """ Obtain a batch of test data
    """
    while True:
      ret = self.get_batch(ProcessType.EVALUATION)
      self.cur_test_index += self.minibatch_size
      if self.cur_test_index > len(self.test_texts) - self.minibatch_size:
        self.cur_test_index = 0
      yield ret

  def load_train_data(self, desc_file='train_corpus.json', cmvn_samples=100):
    self.load_metadata_from_desc_file(desc_file, ProcessType.TRAINING)
    if self.feats_mean is None or self.feats_std is None:
      self.feats_mean, self.feats_std = self.fit_train(cmvn_samples)
      cmvn_mean_file = desc_file.replace(".json", "") + "_" + \
                       self.feat_type + ".mean"
      cmvn_std_file = desc_file.replace(".json", "") + "_" + \
                      self.feat_type + ".std"
      np.savetxt(cmvn_mean_file, self.feats_mean)
      np.savetxt(cmvn_std_file, self.feats_std)
      self.logger.info("mean: %s", cmvn_mean_file)
      self.logger.info("std: %s", cmvn_std_file)

  def load_validation_data(self, desc_file='valid_corpus.json'):
    self.load_metadata_from_desc_file(desc_file, ProcessType.VALIDATION)

  def load_test_data(self, desc_file='test_corpus.json'):
    self.load_metadata_from_desc_file(desc_file, ProcessType.EVALUATION)

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
        except json.decoder.JSONDecodeError as err: #pylint: disable=no-member
          # json module version
          self.logger.error('Error reading line #{}: {}, {}'
                            .format(line_num, json_line, err.msg))
    if partition == ProcessType.TRAINING:
      self.train_audio_paths = audio_paths
      self.train_durations = durations
      self.train_texts = texts
    elif partition == ProcessType.VALIDATION:
      self.valid_audio_paths = audio_paths
      self.valid_durations = durations
      self.valid_texts = texts
    elif partition == ProcessType.EVALUATION:
      self.test_audio_paths = audio_paths
      self.test_durations = durations
      self.test_texts = texts
    else:
      raise Exception("Invalid partition to load metadata. Must be %s/%s/%s"
                      % (ProcessType.TRAINING, ProcessType.VALIDATION,
                         ProcessType.EVALUATION))

  def fit_train(self, k_samples=100):
    """ Estimate the mean and std of the features from the training set
    Params:
      k_samples (int): the number of samples to compute mean and std over feats.
    """
    assert k_samples != 0

    if k_samples == -1:
      k_samples = len(self.train_audio_paths)
    else:
      k_samples = min(k_samples, len(self.train_audio_paths))

    self.logger.info("CMVNs are being computed for %d utterances.", k_samples)
    samples = self.rng.sample(self.train_audio_paths, k_samples)
    feats = [self.featurize(s) for s in samples]
    feats = np.vstack(feats)
    return np.mean(feats, axis=0), np.std(feats, axis=0)

  def graves_2012(self, wav_path):
    """
    Alex. Graves:
    Sequence Transduction with Recurrent Neural Networks.
    CoRR abs/1211.3711 (2012)

    MFCC features
    Standard speech preprocessing was applied to transform the audio files into
    feature sequences. 26 channel mel-frequency filter bank and a pre-emphasis
    coefficient of 0.97 were used to compute 12 mel-frequency cepstral coeffici-
    ents plus an energy coefficient on 25ms Hamming windows at 10ms intervals.
    Delta coefficients were added to create input sequences of length 26 vectors

    For CMVN
    and all coefficient were normalised to have mean zero and standard deviat-
    ion one over the train- ing set. ==> please set --prep-cmvn-samples to -1.

    I left as default the other options which were not mentioned in the paper
    such as nfft, lowfreq, highfreq, ceplifter, etc.

    :param wav_path: wav file path
    :return: a feature sequence
    """
    (rate, sig) = wav.read(Util.get_file_path(self.basepath, wav_path))
    # computing features
    mfcc_feat = \
      mfcc(signal=sig, samplerate=rate, numcep=12, winlen=0.025, nfilt=26,
           winstep=0.01, preemph=0.97, appendEnergy=False, winfunc=np.hamming)
    # adding energy
    energy = np.expand_dims(np.sum(np.power(mfcc_feat, 2), axis=-1), 1)
    mfcc_e_feat = np.concatenate((energy, mfcc_feat), axis=-1)
    # concatenating a delta vector
    delta_feat = delta(mfcc_e_feat, 1)
    return np.concatenate((mfcc_e_feat, delta_feat), axis=1)

  def graves_2013(self, wav_path):
    """
    Alex Graves, Abdel-rahman Mohamed, Geoffrey E. Hinton:
    Speech recognition with deep recurrent neural networks.
    ICASSP 2013: 6645-6649

    FBANK features : (40 fbank, 1 energy * 3)
    The audio data was encoded using a Fourier-transform-based filter-bank with
    40 coefficients (plus energy) distributed on a mel-scale, together with their
    first and second temporal derivatives. Each input vector was therefore size 123.

    For CMVN
    The data were normalised so that every element of the input vec- tors had
    zero mean and unit variance over the training set.

    there is not description about window I chose to use a hanning window.

    I left as default the other options which were not mentioned in the paper
    such as nfft, lowfreq, highfreq, ceplifter, etc.

    :param wav_path: wav file path
    :return: a feature sequence
    """
    (rate, sig) = wav.read(Util.get_file_path(self.basepath, wav_path))
    # computing features
    fbank_feat, _ = \
      fbank(signal=sig, samplerate=rate, nfilt=40, winfunc=np.hanning)

    # adding energy
    energy = np.expand_dims(np.sum(np.power(fbank_feat, 2), axis=-1), 1)
    fbank_e_feat = np.concatenate((energy, fbank_feat), axis=-1)
    # concatenating delta vectors
    delta_feat = delta(fbank_e_feat, 1)
    delta_delta_feat = delta(fbank_e_feat, 2)
    return np.concatenate((fbank_e_feat, delta_feat, delta_delta_feat), axis=1)

  def featurize(self, wav_path):
    """ For a given audio clip, calculate the corresponding feature
    Params:
      audio_clip (str): Path to the audio clip
    """
    if self.feat_type == Constants.FEAT_GRAVES12:
      return self.graves_2012(wav_path)
    elif self.feat_type == Constants.FEAT_GRAVES13:
      return self.graves_2013(wav_path)
    elif self.feat_type == Constants.FEAT_FBANK:
      fbank_feat = \
        Util.get_fbanks(Util.get_file_path(self.basepath, wav_path),
                        frame_size=0.025, frame_stride=0.01,
                        n_filt=self.feat_dim)
      return fbank_feat

    self.logger.error("%s is not supported yet.", self.feat_type)
    sys.exit(1)

  def normalize(self, feature, eps=1e-14):
    """ Center a feature using the mean and std
    Params:
      feature (numpy.ndarray): Feature to normalize
    """
    return (feature - self.feats_mean + eps) / (self.feats_std + eps)

class AudioGeneratorForRNNT(AudioGenerator):
  def __init__(self, logger, config, vocab):
    super().__init__(logger, config, vocab)

  def get_batch(self, partition):
    """ Obtain a batch of train, validation, or test data

      Input for Predicition Network
      The length U + 1 input sequence yˆ = (∅, y1, . . . , yU) to G output
      sequence y with ∅ prepended.

      Label is not prepended. only input sequence.

      Prediction networks lean
      ∅ -> y1
      y1 -> y2
      ...

      In this set-up, the last token in the input will be ignored.

      :param partition: the type of this batch
      :return: input data for rnnt
    """
    if partition == ProcessType.TRAINING:
      audio_paths = self.train_audio_paths
      cur_index = self.cur_train_index
      texts = self.train_texts
    elif partition == ProcessType.VALIDATION:
      audio_paths = self.valid_audio_paths
      cur_index = self.cur_valid_index
      texts = self.valid_texts
    elif partition == ProcessType.EVALUATION:
      audio_paths = self.test_audio_paths
      cur_index = self.cur_test_index
      texts = self.test_texts
    else:
      raise Exception("Invalid partition. Must be %s/%s"%
                      (ProcessType.TRAINING, ProcessType.VALIDATION))

    # extracting features
    features = [self.normalize(self.featurize(a)) for a in
                audio_paths[cur_index:cur_index+self.minibatch_size]]

    # calculate necessary sizes
    batch_size = min(len(features), self.minibatch_size)
    max_feat_len = max([features[i].shape[0] for i in range(0, batch_size)])

    # plus two for BOS and EOS
    if self.is_char:
      max_stri_len = max([len(texts[cur_index + i])
                          for i in range(0, batch_size)])
    else:
      max_stri_len = max([len(texts[cur_index + i].split(" "))
                          for i in range(0, batch_size)])

    # initialize the arrays
    # Input for each network: blank_symbol prepended.
    input_tran = np.zeros([batch_size, max_feat_len, self.feat_dim])
    input_pred = np.zeros([batch_size, max_stri_len + 1, len(self.vocab)])

    # Input for computing rnnt losses
    label_rnnt = np.zeros([batch_size, max_stri_len])
    input_length = np.zeros([batch_size])
    label_length = np.zeros([batch_size])

    for i in range(0, batch_size):
      # Input for Transcription network
      feat = features[i]
      input_length[i] = feat.shape[0] # T
      input_tran[i, :feat.shape[0], :] = feat # x


      int_seq = Util.get_int_seq(texts[cur_index + i], self.is_char, self.vocab)
      label_length[i] = len(int_seq) # U
      label_rnnt[i, :len(int_seq)] = np.array(int_seq) # => U

      # A blank symbol is prepended to the input of a prediction network
      for elem_idx, elem in enumerate(int_seq):
        input_pred[i, elem_idx + 1, elem] = 1

    # return the arrays
    inputs = {Constants.INPUT_TRANS: input_tran,   # [B, T, F]
              Constants.INPUT_PREDS: input_pred,   # [B, U + 1, V]
              Constants.INPUT_INLEN: input_length, # [B] : T
              Constants.INPUT_LBLEN: label_length, # [B] : U
              Constants.INPUT_LABEL: label_rnnt    # [B, U]
             }

    outputs = {Constants.LOSS_RNNT: np.zeros([batch_size])}

    return inputs, outputs

class AudioGeneratorForCTC(AudioGenerator):
  def __init__(self, logger, config, vocab):
    super().__init__(logger, config, vocab)

  def get_batch(self, partition):
    """ Obtain a batch of train, validation, or test data
    """
    if partition == ProcessType.TRAINING:
      audio_paths = self.train_audio_paths
      cur_index = self.cur_train_index
      texts = self.train_texts
    elif partition == ProcessType.VALIDATION:
      audio_paths = self.valid_audio_paths
      cur_index = self.cur_valid_index
      texts = self.valid_texts
    elif partition == ProcessType.EVALUATION:
      audio_paths = self.test_audio_paths
      cur_index = self.cur_test_index
      texts = self.test_texts
    else:
      raise Exception("Invalid partition. Must be %s/%s"%
                      (ProcessType.TRAINING, ProcessType.VALIDATION))

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
    inputs = {Constants.INPUT_TRANS: x_data,
              Constants.INPUT_LABEL: labels,
              Constants.INPUT_INLEN: input_length,
              Constants.INPUT_LBLEN: label_length
             }

    outputs = {Constants.LOSS_CTC: np.zeros([batch_size])}

    return inputs, outputs
