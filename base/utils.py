# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments, too-many-locals

import numpy as np
from scipy.io import wavfile # reading the wavfile
from scipy.fftpack import dct

from base.common import Constants

class KmRNNTUtil:
  @staticmethod
  def mfcc_features(path_file, frame_size=0.025, frame_stride=0.01):
    """
    I borrowed this feature extraction code from
    https://www.kaggle.com/ybonde/log-spectrogram-and-mfcc-filter-bank-example
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    :param path_file: path for a speech file
    :param frame_size: the length for frame (default = 0.05, it means 50 ms)
    :param frame_stride: the length for striding (default = 0.03, it means 30 ms)
    :return:
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

    # hamming window
    frames *= np.hamming(frame_length)

    n_fft = 512
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))  # Magnitude of the FFT
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))  # Power Spectrum

    n_filt = 40
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

    num_ceps = 20
    # Keep 2-13
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]

    cep_lifter = 22
    (_, n_coeff) = mfcc.shape
    coeff_arr = np.arange(n_coeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * coeff_arr / cep_lifter)
    mfcc *= lift  # *

    return filter_banks, mfcc

  @staticmethod
  def load_vocab(path, is_char=True, is_bos_eos=False):
    """
    This method follows the blank symbol policy

    url: https://kite.com/python/docs/keras.backend.ctc.ctc_loss

    The `inputs` Tensor's innermost dimension size, `num_classes`, represents
    `num_labels + 1` classes, where num_labels is the number of true labels, and
    the largest value `(num_classes - 1)` is reserved for the blank label.

    For example, for a vocabulary containing 3 labels `[a, b, c]`,
    `num_classes = 4` and the labels indexing is `{a: 0, b: 1, c: 2, blank: 3}`.

    :param path: vocabulary file
    :param is_char: the unit of vocabulary is a word or a character?
    :param is_bos_eos: begin of sentences and end of sentences need to be added
    :return:
    """
    word_to_id = dict()
    id_to_word = list()

    # bos and eos will be needed ?
    if is_bos_eos:
      id_to_word.append(Constants.BOS)
      id_to_word.append(Constants.EOS)

    with open(path) as file:
      for line in file:
        line = line.strip()
        len_line = len(line)
        if line and len_line > 0:
          id_to_word.append(line.split(" ")[0])

    if is_char:
      id_to_word.append(Constants.SPACE)

    # reserve for a blank symbol
    id_to_word.append(Constants.BLANK)

    # for making a word dictionary
    for i, word in enumerate(id_to_word):
      word_to_id[word] = i

    return word_to_id, id_to_word
