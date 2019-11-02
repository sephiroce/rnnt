# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods, too-many-locals, no-member,
# pylint: disable=too-many-statements, no-name-in-module, import-error

"""rnnt_decoder.py: RNN-T beam search decoder"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import copy
import math
import sys
import numpy as np

from keras.models import model_from_json, Model
from keras.optimizers import SGD

from rnnt.base.util import Util
from rnnt.base.common import Constants, Logger, ParseOption
from rnnt.base.data_generator import AudioGeneratorForRNNT


class Sequence(object):
  """
  class for storing sequence information
  """

  def __init__(self, blank, seq=None):
    if seq is None:
      self.dist = []  # predictions of phoneme language model
      self.hyp = [blank]  # prediction phoneme label
      self.logp = 0  # probability of this sequence, in log scale
    else:
      self.dist = seq.g[:]  # save for prefixsum
      self.hyp = seq.k[:]
      self.logp = seq.logp

class KerasRNNTDecoder(object):
  def __init__(self, vocab, config):
    self.vocab_size = len(vocab)
    self.blank = self.vocab_size
    self.vocab = vocab

    with open(Util.get_file_path(config.paths_data_path,
                                 config.paths_model_json)) as json_file:
      model = model_from_json(json_file.read())
      model.load_weights(Util.get_file_path(config.paths_data_path,
                                            config.paths_model_h5))
      input_tran = None
      input_pred = None
      output_tran = None
      output_pred = None
      model.summary()
      for layer in model.layers:
        if layer.name == Constants.INPUT_TRANS:
          input_tran = layer.input
        if layer.name == Constants.INPUT_PREDS:
          input_pred = layer.input
        if layer.name == Constants.OUTPUT_TRANS:
          output_tran = layer.output
        if layer.name == Constants.OUTPUT_PREDS:
          output_pred = layer.output

      self.encoder = Model(inputs=input_tran, outputs=output_tran, name="encoder")
      self.decoder = Model(inputs=input_pred, outputs=output_pred, name="decoder")
      self.encoder.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.0))
      self.decoder.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.0))
      self.encoder.summary()
      self.decoder.summary()

  def beam_search(self, feature_seq, beam_size=10): # pylint: disable=too-many-branches
    """

    :param feature_seq: acoustic model outputs
    :param beam_size: beam size
    :return: a decoding result
    """

    # RNN Layers to encode a speech signal sequence
    # precomputed Pr({k U blank} | y,t)
    encoder_output_sequence = np.squeeze(self.encoder.predict(np.array([feature_seq])))

    # LINE1: Initialize: B={blank}; , Where is Pr(blank) = 1 => in Sequence class
    B = [Sequence(blank=self.blank)] # pylint: disable=invalid-name

    # LINE2: for t = 1 to T do
    for encoder_output_t in encoder_output_sequence: # pylint: disable=too-many-nested-blocks
      # larger sequence first add
      sorted(B, key=lambda a: len(a.k), reverse=True)

      # A: emitting non blank
      # B: Non-emitting? anyway blank!
      # LINE3: A = B
      A = B # pylint: disable=invalid-name
      # LINE4: B = {}
      B = [] # pylint: disable=invalid-name

      # LINE5: for y in A do
      # for j in range(len(A) - 1):
      for y_idx, y in enumerate(A):  # pylint: disable=invalid-name
        if y_idx == len(A) - 1:
          break
        # LINE6: Pr(y) += sum_hat(y)_in_prefix(y) ∩ A Pr(hat{y})Pr(y|hat{y}, t)
        # for i in range(y_idx + 1, len(A)):
        # sum
        for y_star in A[y_idx + 1:]:
          # hat(y)_in_prefix(y) ∩ A
          # a = y_hat.h
          # b = y.k
          # y_hat is in y prefix set intersected with A
          if y_star.k == y.k or len(y_star.k) >= len(y.k):
            continue
          else:
            is_prefix = True
            for idx in range(len(y_star.k)):
              if y_star.k[idx] != y.k[idx]:
                is_prefix = False
                break

            if not is_prefix:
              continue

          # Pr(y|hat{y}, t)
          # Pr(y|y_hat, t)
          cur_seq = np.array([Util.one_hot(y_star.k, self.vocab_size)])
          pred = np.squeeze(self.decoder.predict(cur_seq)[:, -1, :])

          idx = len(y_star.k)
          logp = Util.softmax(pred + encoder_output_t, is_log=True)
          # current log prob = Pr(y_hat) * Pr(y|y_hat,t)
          curlogp = y_star.logp + float(logp[y.k[idx]])
          # sum
          for k_idx in range(idx, len(y.k) - 1):
            logp = Util.softmax(y.g[k_idx] + encoder_output_t, is_log=True, axis=0)
            curlogp += float(logp[y.k[k_idx + 1]].asscalar())
          y.logp = max(y.logp, curlogp) + math.log1p(math.exp(-math.fabs(y.logp - curlogp)))
      # LINE7: end for

      # LINE8: while B contains less than W elements more probable than the most probable in A:
      while True:
        # LINE9: y* = most probable in A
        y_star = max(A, key=lambda a: a.logp)

        # B contains less than W elements more probable than the most probable in A
        if B:
          if len(B) < beam_size and max(B, key=lambda a: a.logp).logp >= y_star.logp:
            pass
          else:
            break

        # LINE10: remove y* from A
        A.remove(y_star)

        # LINE11: Pr(y_star)Pr({blank, vocabs}|y,t)
        # calculate P(k|y_hat, t) for all k including a blank symbol
        #pred, hidden = self.forward_step(y_star.k[-1], y_star.h) # get last label and hidden state
        cur_seq = np.array([Util.one_hot(y_star.k, self.vocab_size)])
        pred = np.squeeze(self.decoder.predict(cur_seq)[:, -1, :])
        logp = Util.softmax(pred + encoder_output_t, is_log=True)

        # LINE13: for k \in vocab: for all vocabs
        for token_id in range(self.vocab_size + 1):
          y_star_plus_k = copy.deepcopy(y_star)

          # Pr(y_star + k) = Pr(y_star)Pr(k|y_star, t)
          y_star_plus_k.logp += float(logp[token_id])

          # LINE12: Add y_star to B
          if token_id == self.blank:
            B.append(y_star_plus_k)  # next move, if blank then, only probability is needed
            continue

          #y_star_plus_k.h = hidden
          y_star_plus_k.k.append(token_id) # word index
          y_star_plus_k.g.append(pred) # prediction of LM

          # Add y_star + k to A
          A.append(y_star_plus_k)
        # LINE14: end for
      # LINE15: end while

      # LINE16: Remove all but the W most probable from B
      sorted(B, key=lambda a: a.logp, reverse=True)
      B = B[:beam_size] # pylint: disable=invalid-name
    # LINE17: end for

    # LINE18: Return: y with highest log Pr(y)/|y| in B
    # return highest probability sequence
    # B[0] <- sorted by logp!, but original algorithm use "Pr(y)/|y|"
    return B[0].hyp, -B[0].logp

def main():
  logger = Logger(name="KerasRNNTDecoder", level=Logger.DEBUG).logger

  # Configurations
  config = ParseOption(sys.argv, logger).args

  # Loading vocabs
  _, vocab = Util.load_vocab(Util.get_file_path(config.paths_data_path,
                                                config.paths_vocab),
                             config=config)
  logger.info("%d words were loaded", len(vocab))
  logger.info("The expanded vocab size : %d", len(vocab) + 1)
  logger.info("The index of a blank symbol: %d", len(vocab))

  # Computing mean and std for cmvn using a AudioGeneratorForRNNT class
  audio_gen = AudioGeneratorForRNNT(logger, config, vocab)
  audio_gen.load_train_data(Util.get_file_path(config.paths_data_path,
                                               "train_corpus.json"), 1000)
  mean = audio_gen.feats_mean
  std = audio_gen.feats_std

  # Decoding
  krd = KerasRNNTDecoder(vocab, config)
  # A wav path is hardcoded.
  speech_path = "samples/data/timit_sample/LDC93S1.wav"
  feature_seq = (audio_gen.featurize(speech_path)- mean) / std

  hyp, logp = krd.beam_search(feature_seq, 4)

  exp_vocab = copy.deepcopy(vocab)
  exp_vocab.append("BLANK")
  print('Prediction: {}\tlog-likelihood {:.2f}\n'.
        format(' '.join([exp_vocab[i] for i in hyp]), -logp))

if __name__ == "__main__":
  main()