# -*- coding: utf-8 -*-

from python_speech_features import logfbank
from base.common import Constants

class KmRNNTUtil:
  @staticmethod
  def get_feats(signal):
    return logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
                    nfilt=40, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)

  @staticmethod
  def load_vocab(path):
    word_to_id = dict()
    id_to_word = list()

    # bos and eos will be needed ?
    id_to_word.append(Constants.BOS)
    id_to_word.append(Constants.EOS)
    with open(path) as file:
      for line in file:
        id_to_word.append(line.strip().split(" ")[0])

    # reserve for a blank symbol
    id_to_word.append(Constants.BLANK)

    # for making a word dictionary
    for i, word in enumerate(id_to_word):
      word_to_id[word] = i

    return word_to_id, id_to_word
