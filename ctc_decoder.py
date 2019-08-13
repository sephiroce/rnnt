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

  json_file = open("results/"+sys.argv[3]+".json", "r")
  loaded_model_json = json_file.read()
  json_file.close()

  model = model_from_json(loaded_model_json)
  model.load_weights("results/"+sys.argv[3]+".h5")
  model.summary()

  audio_gen = AudioGenerator(logger,
                             basepath=basepath,
                             vocab=vocab,
                             minibatch_size=1,
                             mfcc_dim=mfcc_dim,
                             max_duration=0, # it means, all duration speeches will be generated.
                             sort_by_duration=False,
                             is_char=True,
                             is_bos_eos=False)

  audio_gen.load_train_data("%s/train_corpus.json" % basepath)
  audio_gen.load_test_data("%s/test_corpus.json" % basepath)

  with open(sys.argv[3]+".utt", "w") as utt_file:
    for i, val in enumerate(audio_gen.next_test()):
      if i == len(audio_gen.test_audio_paths):
        break
      result = KMCTC.get_result_str(model.predict(val[0]), id_to_word)
      logger.info("UTT%03d: %s", i+1, result)
      utt_file.write("%s (spk-%d)\n"%(result, i+1))
  logger.info("UTT File saved into %s.utt", sys.argv[3])

if __name__ == "__main__":
  main()
