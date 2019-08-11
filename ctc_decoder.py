# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals,

import sys

from keras.models import model_from_json
from base.utils import KmRNNTUtil as Util
from base.common import Logger
from base.data_generator import AudioGenerator
from ctc import KMCTC

def main():
  logger = Logger(name="KmRNNT_CTC_Decoder", level=Logger.DEBUG).logger

  # Options
  basepath = sys.argv[1]
  mfcc_dim = 20

  vocab, id_to_word = Util.load_vocab(sys.argv[2], is_char=True, is_bos_eos=False)

  json_file = open("results/inference.mfcc20.layer2_300.json", "r")
  loaded_model_json = json_file.read()
  json_file.close()

  model = model_from_json(loaded_model_json)
  model.load_weights("results/inference.mfcc20.layer2_300.h5")
  model.summary()

  audio_gen = AudioGenerator(logger,
                             basepath=basepath,
                             vocab=vocab,
                             minibatch_size=1,
                             mfcc_dim=mfcc_dim,
                             max_duration=99999,
                             sort_by_duration=False,
                             is_char=True,
                             is_bos_eos=False)

  audio_gen.load_train_data("%s/train_corpus.json" % basepath)
  audio_gen.load_test_data("%s/test_corpus.json" % basepath)

  for i, val in enumerate(audio_gen.next_test()):
    if i == len(audio_gen.test_audio_paths):
      break
    result = KMCTC.get_result_str(model.predict(val[0]), id_to_word)
    print("UTT%03d: %s"%(i+1, result))

if __name__ == "__main__":
  main()
