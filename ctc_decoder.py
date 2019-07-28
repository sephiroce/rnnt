# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals,

import os
import sys

from base.utils import KmRNNTUtil as Util
from base.common import Logger
from base.data_generator import AudioGenerator
from ctc import KMCTC as km_ctc

def main():
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  logger = Logger(name="KmRNNT_CTC_Decoder", level=Logger.DEBUG).logger

  # Options
  basepath = sys.argv[1]
  feat_dim = 40
  minibatch_size = 100
  sort_by_duration = False
  max_duration = 50.0

  vocab, _ = Util.load_vocab(sys.argv[2], is_char=True, is_bos_eos=False)

  model, _, _ = \
  km_ctc.create_model(input_dim=feat_dim, output_dim=len(vocab))

  model.load_weights("results/inference")
  model.summary()

  audio_gen = AudioGenerator(logger, basepath=basepath, vocab=vocab,
                             minibatch_size=minibatch_size, feat_dim=feat_dim,
                             max_duration=max_duration,
                             sort_by_duration=sort_by_duration,
                             is_char=True, is_bos_eos=False)

  # add the training data to the generator
  audio_gen.load_test_data("%s/test_corpus.json"%basepath)
  y_pred_proba = model.predict_generator(generator=audio_gen.next_test(),
                                         steps=1, verbose=1)
  logger.info(y_pred_proba)
if __name__ == "__main__":

  main()
