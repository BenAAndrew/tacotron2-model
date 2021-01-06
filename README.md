# tacotron2-model
A PyPI port of the NVIDIA Tacotron2 model

Source: https://github.com/NVIDIA/tacotron2 (model.py)

A pytorch install is required but is not added to requirements to avoid configuration issues.

The only change from the NVIDIA original is a replacement of hparams with individual arguments.
This removes the dependency on tf.contrib.training.HParams (deprecated since tensorflow 1).

## Model usage

```
from tacotron2_model import Tacotron2

model = Tacotron2().cuda()
print(model.eval())
```

## Loss usage

```
from tacotron2_model import Tacotron2Loss

criterion = Tacotron2Loss()
```


## Collate usage

```
from tacotron2_model import TextMelCollate

collate_fn = TextMelCollate()
```
