# tacotron2-model
A PyPI port of the NVIDIA Tacotron2 model

Source: https://github.com/NVIDIA/tacotron2 (model.py)

Only change from the NVIDIA original is a replacement of hparams with individual arguments to remove the dependency on tf.contrib.training.HParams (deprecated since tensorflow 1).
