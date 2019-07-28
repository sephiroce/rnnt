# -*- coding: utf-8 -*-
# pylint: disable=too-many-arguments

from python_speech_features import logfbank
from python_speech_features import mfcc
import soundfile as sf
import numpy as np

from base.common import Constants

class KmRNNTUtil:
  @staticmethod
  def get_logfbank(path, feat_dim):
    signal, _ = sf.read(path)
    return logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
                    nfilt=feat_dim, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)

  @staticmethod
  def get_mfcc(signal, numcep):
    return mfcc(signal, 16000, numcep=numcep)

  @staticmethod
  def load_vocab(path, is_char=False, is_bos_eos=True):
    word_to_id = dict()
    id_to_word = list()

    # bos and eos will be needed ?
    if is_bos_eos:
      id_to_word.append(Constants.BOS)
      id_to_word.append(Constants.EOS)

    with open(path) as file:
      for line in file:
        id_to_word.append(line.strip().split(" ")[0])

    if is_char:
      id_to_word.append(Constants.SPACE)

    # reserve for a blank symbol
    id_to_word.append(Constants.BLANK)

    # for making a word dictionary
    for i, word in enumerate(id_to_word):
      word_to_id[word] = i

    return word_to_id, id_to_word

  @staticmethod
  def get_cmvn(wav_list, feat_type, feat_dim, mean="mean", std="variance", basepath="."):
    feats = []
    with open(wav_list) as f_wav_list:
      for path in f_wav_list:
        if feat_type == Constants.LOG_FBANK:
          feats.append(KmRNNTUtil.get_logfbank("%s/%s"%(basepath, path.strip()), feat_dim))
    feats = np.vstack(feats)
    np.save(mean, np.mean(feats, axis=0))
    np.save(std, np.std(feats, axis=0))
