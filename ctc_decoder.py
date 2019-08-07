# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals,

import os
import sys

from base.utils import KmRNNTUtil as Util
from base.common import Logger
from base.data_generator import AudioGenerator
from ctc import KMCTC as km_ctc
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
from keras import backend as k 

def main():
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  logger = Logger(name="KmRNNT_CTC_Decoder", level=Logger.DEBUG).logger

  # Options
  basepath = sys.argv[1]
  feat_dim = 40
  minibatch_size = 100
  sort_by_duration = False
  max_duration = 50.0

  vocab, id_to_word = Util.load_vocab(sys.argv[2], is_char=True, is_bos_eos=False)

  json_file = open("results/inference.new.json", "r")
  loaded_model_json = json_file.read()
  json_file.close()

  model = model_from_json(loaded_model_json)
  model.load_weights("results/inference.new.h5")
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
  input_length = list()
  for y in y_pred_proba:
    input_length.append(len(y))

  results = \
    tf.keras.backend.ctc_decode(y_pred_proba,
            input_length = input_length, 
            greedy=True, 
            beam_width=12,
            top_paths=1)
  sess = tf.Session()
  with sess.as_default():
    results_arr = k.get_value(results[0][0])
    for sent_arr in results_arr:
      sent=""
      for char in sent_arr:
        if char < 0 :
          break
        if char == len(vocab) - 1:
          sent += " "
        else:
          sent += id_to_word[char]
      print(sent)

if __name__ == "__main__":
  main()
