# tacotron2-model
A PyPI port of the NVIDIA Tacotron2 model

Source: https://github.com/NVIDIA/tacotron2 (model.py)

A pytorch install is required but is not added to requirements to avoid configuration issues.

The only change from the NVIDIA original is a replacement of hparams with individual arguments.
This removes the dependency on tf.contrib.training.HParams (deprecated since tensorflow 1).

## Model usage

```
from tacotron2_model import Tacotron2

model = Tacotron2(N_MEL_CHANNELS, N_SYMBOLS, SYMBOLS_EMBEDDING_DIM, 
                  ENCODER_N_CONVOLUTIONS, ENCODER_EMBEDDING_DIM, 
                  ENCODER_KERNEL_SIZE, ATTENTION_RNN_DIM, ATTENTION_DIM,
                  ATTENTION_LOCATION_N_FILTERS, ATTENTION_LOCATION_KERNEL_SIZE,
                  DECODER_RNN_DIM, PRENET_DIM, MAX_DECODER_STEPS, GATE_THRESHOLD,
                  P_ATTENTION_DROPOUT, P_DECODER_DROPOUT, POSTNET_EMBEDDING_DIM,
                  POSTNET_KERNEL_SIZE, POSTNET_N_CONVOLUTIONS).cuda()

print(model.eval())
```

## Loss usage

```
from tacotron2_model import Tacotron2Loss

criterion = Tacotron2Loss()
```

## Example params

```
N_MEL_CHANNELS = 80
N_SYMBOLS = 148
SYMBOLS_EMBEDDING_DIM = 512

ENCODER_N_CONVOLUTIONS = 3
ENCODER_EMBEDDING_DIM = 512
ENCODER_KERNEL_SIZE = 5

ATTENTION_RNN_DIM = 1024
ATTENTION_DIM = 128
ATTENTION_LOCATION_N_FILTERS = 32
ATTENTION_LOCATION_KERNEL_SIZE = 31

DECODER_RNN_DIM = 1024
PRENET_DIM = 256
MAX_DECODER_STEPS = 1000
GATE_THRESHOLD = 0.5

P_ATTENTION_DROPOUT = 0.1
P_DECODER_DROPOUT = 0.1

POSTNET_EMBEDDING_DIM = 512
POSTNET_KERNEL_SIZE = 5
POSTNET_N_CONVOLUTIONS = 5 
```
