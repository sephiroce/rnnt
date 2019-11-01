# Keras based RNN-T
## How to use
* To train models. If decoder options are not set, then a model will be trained using CTC loss.
 
```
#!/bin/bash

export PYTHONPATH=`pwd`
python -u rnnt/train.py \
--paths-data-path=${DATA} \ # This toolkit supports memory based generators, I checked wsj and timit.
--config=${CONFIG_FILE} \ # please check sample configuration files in ./samples/conf/*.conf
--paths-vocab=${TOKEN_LIST_FILE} # row number - 1 is word index, a blank symbol is represented by the last row number.
```

## Data preparation
* json format
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