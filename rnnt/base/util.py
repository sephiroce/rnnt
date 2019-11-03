# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments, too-many-locals, import-error

"""util.py: Utilities for ASR systems"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
from rnnt.base.common import Constants, ExitCode

class Util:
  @staticmethod
  def get_int_seq(text, is_char, vocab):
    int_seq = list()

    if is_char:
      for char in text.strip():
        if char in vocab:
          int_seq.append(vocab[char])
        elif char == ' ':
          int_seq.append(vocab[Constants.SPACE])
        else:
          sys.exit(ExitCode.NOT_SUPPORTED)
    else:
      for bpe in text.strip().split(" "):
        if bpe in vocab:
          int_seq.append(vocab[bpe])
        else:
          int_seq.append(vocab[Constants.UNK])
    return int_seq

  @staticmethod
  def get_file_path(data_path, file_path):
    """
    In order to handle both absolute paths and relative paths, it returns an
    exist path among combinations of data_path and file_path.

    :param data_path: base path
    :param file_path: file path
    :return: a existed file path
    """
    data_path = data_path.strip()
    file_path = file_path.strip()
    return file_path if os.path.isfile(file_path) \
      else data_path + "/" + file_path

  @staticmethod
  def rnnt_lambda_func(args):
    y_trans, y_pred, labels, input_length, label_length = args
    import keras.backend as K
    import tensorflow as tf

    # the 2 is critical here since the first couple outputs of the RNN tend to
    # be garbage:
    shift = 2
    y_trans = y_trans[:, shift:, :]
    input_length -= shift

    # calculating lattices from the output from the prediction network and
    # the transcription network.
    batch_size = K.shape(y_trans)[0]
    y_trans = K.expand_dims(y_trans, axis=2) # BT1H
    y_pred = K.expand_dims(y_pred, axis=1) # B1UH
    acts = tf.nn.log_softmax(y_trans + y_pred)
    input_length = K.reshape(input_length, [batch_size])
    label_length = K.reshape(label_length, [batch_size])

    from warprnnt_tensorflow import rnnt_loss
    list_value = rnnt_loss(acts, labels, input_length, label_length,
                           blank_label=39)

    return tf.reshape(list_value, [batch_size])

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
  def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    shift = 2
    y_pred = y_pred[:, shift:, :]
    input_length -= shift

    from keras import backend as k
    return k.ctc_batch_cost(labels, y_pred, input_length, label_length)

  @staticmethod
  def get_fbanks(path_file, frame_size=0.025, frame_stride=0.01, n_filt=40, num_ceps=-1):
    """
    I borrowed this feature extraction code from
    https://www.kaggle.com/ybonde/log-spectrogram-and-mfcc-filter-bank-example
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine
    -learning.html
    :param n_filt: applying triangular n_filt filsters filters, typically 40 filters,
      nfilt = 40 on a Mel-scale to the power spectrum
      to extract frequency bands. The Mel-scale aims to mimic the non-linear human ear
      perception of sound, by being more discriminative at lower frequencies and less
      discriminative at higher frequencies. The final step to computing filter banks is
       applying triangular filters, typically 40 filters, nfilt = 40 on a Mel-scale to
       the power spectrum to extract frequency bands. The Mel-scale aims to mimic the
       non-linear human ear perception of sound, by being more discriminative at lower
        frequencies and less discriminative at higher frequencies.
    :param path_file: path for a speech file
    :param frame_size: the length for frame (default = 0.05, it means 50 ms)
    :param frame_stride: the length for striding (default = 0.03, it means 30
    ms)
    :param num_ceps: mfcc dimension, if it is bigger than 0 then it returns mfcc
    :return: fbank features
    """
    sample_rate, signal = wavfile.read(path_file)
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0],
                                  signal[1:] - pre_emphasis * signal[:-1])

    # params
    # Convert from seconds to samples
    frame_length, frame_step = frame_size * sample_rate, frame_stride * \
                               sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) /
                             frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z_value = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(emphasized_signal, z_value)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step),
                      (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # applying hamming window to all frames
    frames *= np.hamming(frame_length)
    # frames *= 0.54 - 0.46 * np.cos((2 * numpy.pi * n) / (frame_length - 1))

    n_fft = 512
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))  # Magnitude of the FFT
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))  # Power Spectrum

    low_freq_mel = 0
    # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filt + 2)
    # Convert Mel to Hz
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bins = np.floor((n_fft + 1) * hz_points / sample_rate)

    fbank = np.zeros((n_filt, int(np.floor(n_fft / 2 + 1))))
    for m_idx in range(1, n_filt + 1):
      f_m_minus = int(bins[m_idx - 1])  # left
      f_m = int(bins[m_idx])  # center
      f_m_plus = int(bins[m_idx + 1])  # right

      for k_idx in range(f_m_minus, f_m):
        fbank[m_idx - 1, k_idx] = (k_idx - bins[m_idx - 1]) / (bins[m_idx] - bins[m_idx - 1])
      for k_idx in range(f_m, f_m_plus):
        fbank[m_idx - 1, k_idx] = (bins[m_idx + 1] - k_idx) / (bins[m_idx + 1] - bins[m_idx])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps,
                            filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    if num_ceps < 0:
      return filter_banks

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
    cep_lifter = 22
    _, ncoeff = mfcc.shape
    ncoeff_arange = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * ncoeff_arange / cep_lifter)
    mfcc *= lift
    return mfcc

  @staticmethod
  def load_vocab(path, config):
    """
    :param path: vocabulary file
    :param config: RNN-LM config
    :return: word_to_id(dic) and id_to_vocab(list)
    """
    word_to_id = dict()
    id_to_word = list()

    with open(path) as file:
      for line in file:
        line = line.strip()
        len_line = len(line)
        if line and len_line > 0:
          id_to_word.append(line.strip())

    # for making a word dictionary
    for i, word in enumerate(id_to_word):
      word_to_id[word] = i

    sanity_check = True
    if len(word_to_id) != len(id_to_word):
      print("duplicated words exit in the vocab file.")
      sanity_check = False

    if config.prep_use_bos and Constants.BOS not in word_to_id:
      print("prep_use_bos is True but no %s in the vocab file."%Constants.BOS)
      sanity_check = False

    if (config.prep_use_eos or config.prep_use_beos) and Constants.EOS not in\
            word_to_id:
      print("prep_use_eos or beos is True but no %s in the vocab file." %
            Constants.EOS)
      sanity_check = False

    if config.prep_text_unit == Constants.CHAR and Constants.SPACE not in \
            word_to_id:
      print("prep_text_unit is %s but no %s in the vocab file." %
            (Constants.CHAR, Constants.SPACE))
      sanity_check = False

    if config.prep_use_unk and Constants.UNK not in word_to_id:
      print("A vocab file must contain unknown symbol \"%s\"." % Constants.UNK)
      sanity_check = False

    if not sanity_check:
      sys.exit(ExitCode.INVALID_DICTIONARY)

    return word_to_id, id_to_word

  @staticmethod
  def softmax(value, is_log=False, axis=None):
    if axis is not None:
      assert axis == 0
      e_x = []
      for elem in value:
        e_x.append(Util.softmax(elem))
    else:
      e_x = np.exp(value)
      if is_log:
        e_x = np.log(e_x) - np.log(e_x.sum())
      else:
        e_x = e_x / e_x.sum()

    return e_x

  @staticmethod
  def one_hot(value, dim):
    """

    :param value: a sequence of token indexes
    :param dim: a dimension of one hot vectors
    :return: a sequence of one hot token vectors
    """

    one_hot_encodeds = []
    for token_id in value:
      one_hot_encoded = np.zeros(dim).tolist()
      # blank labels can be represented by zero vectors
      if token_id < dim:
        one_hot_encoded[token_id] = 1
      one_hot_encodeds.append(one_hot_encoded)
    return one_hot_encodeds
