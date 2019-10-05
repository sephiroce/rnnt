import sys

import os
from keras.layers import Bidirectional, Activation
from base.common import Constants, Logger, ParseOption, ExitCode
from base.utils import KmRNNTUtil as Util
from keras.layers import Dense, TimeDistributed, Input, CuDNNLSTM, Lambda
from keras.models import Model
from keras.optimizers import SGD
from base.data_generator_rnnt import AudioGeneratorForRNNT

cell_size = 500
layers = 5
epochs = 300
minibatch_size = 24
lr = 0.00001

class KMRNNT(object):
  @staticmethod
  def create_model(config, input_dim, vocab, gpus=1):
    """
    this method creates a ctc model
    please modify this method directly just like setting hyper-parameters for
    the models.
    :param input_dim: feat_dim
    :param vocab:
    :param gpus: the number of gpus
    :return: a ctc model
    """

    # Five Inputs
    input_tran = Input(name=Constants.INPUT_TRANS, shape=(None, input_dim))
    input_pred = Input(name=Constants.INPUT_PREDS, shape=(None, len(vocab)))
    input_length = Input(name=Constants.INPUT_INLEN, shape=[1], dtype='int32')
    label_length = Input(name=Constants.INPUT_LBLEN, shape=[1], dtype='int32')
    labels = Input(name=Constants.INPUT_LABEL, shape=[None], dtype='int32')

    # CTC model: Bidirectional LSTM
    prev_layer = Bidirectional(CuDNNLSTM(config.model_layer_size,
                                         return_sequences=True),
                               merge_mode='concat')(input_tran)
    y_trans = TimeDistributed(Dense(len(vocab) + 1))(prev_layer)

    # RNN Language Model: Unidirectional LSTM
    prev_layer = CuDNNLSTM(config.model_layer_size,
                           return_sequences=True)(input_pred)
    y_pred = TimeDistributed(Dense(len(vocab) + 1))(prev_layer)

    # declaring lambda function
    loss_out = Lambda(Util.rnnt_lambda_func, output_shape=(1,),
                      name=Constants.LOSS_RNNT)\
      ([y_trans, y_pred, labels, input_length, label_length])

    # creating a model
    model = Model(inputs=[input_tran, input_pred, labels, input_length,
                          label_length],
                  outputs=loss_out)

    # multi gpu
    if gpus >= 2:
      from keras.utils.training_utils import multi_gpu_model
      model = multi_gpu_model(model, gpus=gpus)

    # compiling a model
    optimizer = SGD(lr=lr, decay=lr * 0.0001, momentum=0.9, nesterov=True,
                    clipnorm=5)

    model.compile(loss={Constants.LOSS_RNNT: lambda y_true, y_pred: y_pred},
                  optimizer=optimizer)

    return model

def main():
  logger = Logger(name="KmRNNT", level=Logger.DEBUG).logger

  # Configurations
  config = ParseOption(sys.argv, logger).args

  # Loading vocabs
  vocab_path = Util.get_file_path(config.paths_data_path, config.paths_vocab)
  if not os.path.isfile(vocab_path):
    logger.critical("%s does not exist.", vocab_path)
    sys.exit(ExitCode.INVALID_FILE_PATH)
  vocab, _ = Util.load_vocab(vocab_path, config=config)
  logger.info("%d words were loaded", len(vocab))
  logger.info("The expanded vocab size : %d", len(vocab) + 1)
  logger.info("The index of a blank symbol: 0")

  model = KMRNNT.create_model(config=config,
                              input_dim=config.feature_dimension,
                              vocab=vocab, gpus=config.device_number_of_gpu)
  model.summary()

  # create a class instance for obtaining batches of data
  audio_gen = AudioGeneratorForRNNT(logger, config, vocab)

  # add the training data to the generator
  audio_gen.load_train_data(Util.get_file_path(config.paths_data_path,
                                               "train_corpus.json"))
  audio_gen.load_validation_data(Util.get_file_path(config.paths_data_path,
                                                    "valid_corpus.json"))
  audio_gen.load_test_data(Util.get_file_path(config.paths_data_path,
                                              "test_corpus.json"))

  # train a new model
  train_batch_size = (len(audio_gen.train_audio_paths)//config.train_batch)
  valid_batch_size = (len(audio_gen.valid_audio_paths)//config.train_batch)

  # fitting a model
  hist = model.fit_generator(generator=audio_gen.next_train(),
                             steps_per_epoch=train_batch_size,
                             epochs=config.train_max_epoch,
                             validation_data=audio_gen.next_valid(),
                             validation_steps=valid_batch_size,
                             verbose=1)

if __name__ == "__main__":
  main()
