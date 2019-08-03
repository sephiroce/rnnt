"""
This is not a completed unit test code.
"""
from base.data_generator import AudioGenerator
from base.utils import KmRNNTUtil as Util
from base.common import Logger

def main():
  vocab, _ = Util.load_vocab("base/test/char.voc", is_char=True,
                             is_bos_eos=False)
  logger = Logger(name="Data_Generator_Test", level=Logger.DEBUG).logger

  audio_gen = AudioGenerator(logger, basepath="base/test", vocab=vocab,
                             minibatch_size=1, feat_dim=40,
                             max_duration=10.0,
                             sort_by_duration=False,
                             is_char=True, is_bos_eos=False)

  # add the training data to the generator
  audio_gen.load_train_data("base/test/test_corpus.json")

  for i, value in enumerate(audio_gen.next_train()):
    if i >= 2:
      break
    logger.info(value)

if __name__ == "__main__":
  main()
