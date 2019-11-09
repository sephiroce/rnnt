# Keras based RNN-T

### Data preparation
* The data generator is a customized version of an AudioGenerator class of lucko515.  
ref: https://github.com/lucko515/speech-recognition-neural-network/blob/master/data_generator.py  
```
Json Format
{
 "duration": length_of_speech_in_secs,
 "text": label,
 "key": a path of a wave file
}
```
* Token list (a list of vocabularies or phones)
```
# a line number is a token index 
phone1
phone2
...
``` 

### Training
* cmvn files are automatically computed and saved unless you set the paths-cmvn-{mean, std}.
```
export PYTHONPATH=`pwd`

python -u rnnt/train.py \
--paths-data-path=$TIMIT_DATA_PATH \
--config=$CONFIG/timit_graves13_CTC-3L-250H.conf \
--paths-vocab=$TIMIT_DATA_PATH/timit_61.phone \
--paths-cmvn-mean=train_core_corpus_61_graves13.mean \
--paths-cmvn-std=train_core_corpus_61_graves13.std \
--paths-train-corpus=train_core_corpus_61.json \
--paths-valid-corpus=valid_noncore_50spk_61.json \
--paths-test-corpus=test_core_corpus_61.json
```

### Decoding
* Same options as train.py have. Just change train.py to ctc_decoder and rnnt_decoder.
