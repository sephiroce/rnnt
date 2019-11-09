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
from keras.layers import Input

from rnnt.base.util import Util
from rnnt.base.common import Constants, Logger, ParseOption
from rnnt.base.data_generator import AudioGeneratorForRNNT

from rnnt.keras_model import KerasModel

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
      self.dist = seq.dist[:]  # save for prefixsum
      self.hyp = seq.hyp[:]
      self.logp = seq.logp

class KerasRNNTDecoder(object):
  def __init__(self, vocab, config):
    """

    :param vocab:
    :param config:
    """
    self.vocab_size = len(vocab)
    self.blank = self.vocab_size

    model_json = Util.get_file_path(config.paths_data_path,
                                    config.paths_model_json)
    model_h5 = Util.get_file_path(config.paths_data_path, config.paths_model_h5)

    with open(model_json) as json_file:
      model = model_from_json(json_file.read())
      model.load_weights(model_h5)
      model.summary()

      input_tran = model.get_layer(Constants.INPUT_TRANS).input
      input_pred = model.get_layer(Constants.INPUT_PREDS).input
      output_tran = model.get_layer(Constants.OUTPUT_TRANS).output
      output_pred = model.get_layer(Constants.OUTPUT_PREDS).output
      self.encoder = \
        Model(inputs=input_tran, outputs=output_tran, name="encoder")
      self.decoder = \
        Model(inputs=input_pred, outputs=output_pred, name="decoder")
      self.encoder.summary()
      self.decoder.summary()

      if config.joint_number_of_layer > 0 and config.joint_layer_size:
        for _ in range(4):
          model.layers.pop()

        for _ in range(5):
          model.layers.pop(0)

        input_joint = Input(name=Constants.INPUT_JOINT,
                            shape=[config.encoder_layer_size * 2 +
                                   config.decoder_layer_size])

        acts = KerasModel.get_fc_decode(input_joint,
                                        len(vocab) + 1,
                                        config.joint_layer_size,
                                        config.joint_number_of_layer)

        self.jointfc = Model(inputs=input_joint, outputs=acts, name="joint")
        self.jointfc.set_weights(model.get_weights())
        self.jointfc.compile(loss="categorical_crossentropy",
                             optimizer=SGD(lr=0.0))
        self.jointfc.summary()
        self.is_joint = True
      else:
        self.is_joint = False

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
      sorted(B, key=lambda a: len(a.hyp), reverse=True)

      # A: emitting non blank
      # B: Non-emitting? anyway blank!
      # LINE3: A = B
      A = copy.deepcopy(B) # pylint: disable=invalid-name
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
          # b = y.hyp
          # y_hat is in y prefix set intersected with A
          if y_star.hyp == y.hyp or len(y_star.hyp) >= len(y.hyp):
            continue
          else:
            is_prefix = True
            for idx in range(len(y_star.hyp)):
              if y_star.hyp[idx] != y.hyp[idx]:
                is_prefix = False
                break

            if not is_prefix:
              continue

          # Pr(y|hat{y}, t)
          # Pr(y|y_hat, t)
          cur_seq = np.array([Util.one_hot(y_star.hyp, self.vocab_size)])
          pred = np.squeeze(self.decoder.predict(cur_seq)[:, -1, :])
          if self.is_joint:
            out = np.concatenate((encoder_output_t, pred), axis=-1)
            out = np.expand_dims(out, 0)
            out = self.jointfc.predict(out)[-1, :]
          else:
            out = pred + encoder_output_t

          logp = Util.softmax(out, is_log=True)

          # current log prob = Pr(y_hat) * Pr(y|y_hat,t)
          idx = len(y_star.hyp)
          curlogp = y_star.logp + float(logp[y.hyp[idx]])
          # sum
          for k_idx in range(idx, len(y.hyp) - 1):
            if self.is_joint:
              out = np.concatenate((encoder_output_t, y.dist[k_idx]), axis=-1)
              out = np.expand_dims(out, 0)
              out = self.jointfc.predict(out)[-1, :]
            else:
              out = y.dist[k_idx] + encoder_output_t
            logp = Util.softmax(out, is_log=True, axis=0)
            curlogp += float(logp[y.hyp[k_idx + 1]])
          y.logp = max(y.logp, curlogp) + math.log1p(math.exp(-math.fabs(y.logp - curlogp)))
      # LINE7: end for

      # LINE8: while B contains less than W elements more probable than the most probable in A:
      while True:
        # LINE9: y* = most probable in A
        y_star = max(A, key=lambda a: a.logp)

        # B contains less than W elements more probable than the most probable in A
        if len(B) >= beam_size and \
                max(B, key=lambda a: a.logp).logp >= y_star.logp:
          break

        # LINE10: remove y* from A
        A.remove(y_star)

        # LINE11: Pr(y_star)Pr({blank, vocabs}|y,t)
        # calculate P(k|y_hat, t) for all k including a blank symbol
        # pred, hidden = self.forward_step(y_star.hyp[-1], y_star.h)
        # in order to get last label and hidden state
        cur_seq = np.array([Util.one_hot(y_star.hyp, self.vocab_size)])
        pred = np.squeeze(self.decoder.predict(cur_seq)[:, -1, :])
        if self.is_joint:
          out = np.concatenate((encoder_output_t, pred), axis=-1)
          out = np.expand_dims(out, 0)
          out = self.jointfc.predict(out)[-1, :]
        else:
          out = pred + encoder_output_t

        logp = Util.softmax(out, is_log=True)

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
          y_star_plus_k.hyp.append(token_id) # word index
          y_star_plus_k.dist.append(pred) # prediction of LM

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
    return B[0].hyp[1:], -B[0].logp

def main():
  logger = Logger(name="KerasRNNTDecoder", level=Logger.DEBUG).logger

  # Configurations
  config = ParseOption(sys.argv, logger).args

  # Loading vocabs
  vocab, id_to_word = Util.load_vocab(Util.get_file_path(config.paths_data_path,
                                                         config.paths_vocab),
                                      config=config)
  logger.info("%d words were loaded", len(vocab))
  logger.info("The expanded vocab size : %d", len(vocab) + 1)
  logger.info("The index of a blank symbol: %d", len(vocab))

  # Computing mean and std for cmvn using a AudioGeneratorForRNNT class
  audio_gen = AudioGeneratorForRNNT(logger, config, vocab)
  if audio_gen.feats_mean is None or audio_gen.feats_std is None:
    audio_gen.fit_train(config.prep_cmvn_samples)
  audio_gen.load_test_data(Util.get_file_path(config.paths_data_path,
                                              config.paths_test_corpus))
  krd = KerasRNNTDecoder(id_to_word, config)
  exp_vocab = copy.deepcopy(id_to_word)
  exp_vocab.append("BLANK")

  if config.inference_is_debug:
    # Decoding
    # A wav path is hardcoded.
    speech_path = "samples/data/timit_sample/LDC93S1.wav"
    feature_seq = (audio_gen.featurize(speech_path) - audio_gen.feats_mean) /\
                  audio_gen.feats_std
    hyp, logp = krd.beam_search(feature_seq, config.inference_beam_width)
    result = ' '.join([exp_vocab[i] for i in hyp])
    logger.info("UTT: %s, %.2f", result, -logp)
  else:
    model_h5 = Util.get_file_path(config.paths_data_path, config.paths_model_h5)
    with open(model_h5 + ".utt", "w") as utt_file:
      for i, val in enumerate(audio_gen.next_test()):
        if i == len(audio_gen.test_audio_paths):
          break
        hyp, logp = krd.beam_search(val[0][Constants.INPUT_TRANS][0],
                                    config.inference_beam_width)
        result = ' '.join([exp_vocab[i] for i in hyp])
        logger.info("UTT%03d: %s, %.2f", i + 1, result, -logp)
        utt_file.write("%s (spk-%d)\n"%(result, i + 1))
      logger.info("A UTT file was saved into %s.utt", model_h5)

if __name__ == "__main__":
  main()
