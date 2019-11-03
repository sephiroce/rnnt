# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals, no-member, import-error, no-name-in-module
# pylint: disable=too-few-public-methods

"""ctc_decoder.py: CTC beam search decoder"""

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"

import sys
import os
import numpy as np

from keras.layers import Lambda, Activation
from keras.models import Model, model_from_json

from rnnt.base.util import Util
from rnnt.base.common import Logger, ParseOption, ExitCode, Constants
from rnnt.base.data_generator import AudioGeneratorForCTC

class KerasCTCDecoder(object):
  @staticmethod
  def ctc_complete_decoding_lambda_func(args, **arguments):
    """
    borrowed from https://github.com/ysoullard/CTCModel/blob/master/CTCModel.py
    :param args:
    :param arguments:
    :return:
    """
    y_pred, input_length = args
    my_params = arguments
    from keras import backend as k
    return k.cast(k.ctc_decode(y_pred, k.squeeze(input_length, 1),
                               greedy=my_params['greedy'],
                               beam_width=my_params['beam_width'],
                               top_paths=my_params['top_paths'])[0][0],
                  dtype='float32')

def main():
  logger = Logger(name="KerasRNNT_CTC_Decoder", level=Logger.DEBUG).logger

  # Configurations
  config = ParseOption(sys.argv, logger).args

  # Loading vocabs
  vocab_path = Util.get_file_path(config.paths_data_path, config.paths_vocab)
  if not os.path.isfile(vocab_path):
    logger.critical("%s does not exist.", vocab_path)
    sys.exit(ExitCode.INVALID_FILE_PATH)
  vocab, id_to_word = Util.load_vocab(vocab_path, config=config)
  logger.info("%d words were loaded.", len(vocab))

  audio_gen = AudioGeneratorForCTC(logger, config, vocab)

  # Loading data
  if audio_gen.feats_mean is None or audio_gen.feats_std is None:
    audio_gen.fit_train(config.prep_cmvn_samples)

  audio_gen.load_test_data(Util.get_file_path(config.paths_data_path,
                                              config.paths_test_corpus))

  # Loading models
  with open(Util.get_file_path(config.paths_data_path,
                               config.paths_model_json), "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model_weight_path = Util.get_file_path(config.paths_data_path,
                                           config.paths_model_h5)
    model.load_weights(model_weight_path)

  if config.inference_is_encoder:
    y_trans = \
      Activation("softmax")(model.get_layer(Constants.OUTPUT_TRANS).output)
  else:
    y_trans = model.get_layer(Constants.OUTPUT_TRANS).output

  if config.inference_is_debug:
    debug_model = Model(inputs=[model.get_layer(Constants.INPUT_TRANS).input],
                        outputs=y_trans)
    debug_model.summary()
    for i, val in enumerate(audio_gen.next_test()):
      if i == 0:
        print(val[0])
        result = debug_model.predict(val[0])
        np.savetxt(model_weight_path + "_%i.csv"%i, result[0], delimiter=",")
        logger.info("Please find a csv file at %s_%d.csv", model_weight_path, i)
        break
  else:
    # Adding a decoder layer to the loaded Keras graph
    out_decoded_dense = \
      Lambda(KerasCTCDecoder.ctc_complete_decoding_lambda_func,
             output_shape=(None, None),
             name=Constants.KEY_CTCDE,
             arguments={'greedy': False,
                        'beam_width': config.inference_beam_width,
                        'top_paths': 1},
             dtype="float32")([y_trans] +
                              [model.get_layer(Constants.INPUT_INLEN).output])

    ctc_model = Model(inputs=[model.get_layer(Constants.INPUT_TRANS).input] + \
                             [model.get_layer(Constants.INPUT_INLEN).input],
                      outputs=out_decoded_dense)
    ctc_model.summary()

    with open(model_weight_path+".utt", "w") as utt_file:
      for i, val in enumerate(audio_gen.next_test()):
        if i == len(audio_gen.test_audio_paths):
          break
        result = Util.get_result_str(ctc_model.predict(val[0]), id_to_word)
        logger.info("UTT%03d: %s", i + 1, result)
        utt_file.write("%s (spk-%d)\n"%(result, i + 1))
      logger.info("A UTT file was saved into %s.utt", model_weight_path)

if __name__ == "__main__":
  main()
