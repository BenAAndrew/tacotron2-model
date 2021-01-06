import pytest
import gdown
import torch

PRETRAINED_WEIGHTS = "https://drive.google.com/uc?id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA"
WEIGHTS_PATH = "tacotron2.pt"


def test_model():
    from tacotron2_model import Tacotron2

    gdown.download(PRETRAINED_WEIGHTS, WEIGHTS_PATH, quiet=False)
    model = Tacotron2()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device("cpu"))["state_dict"])
    _ = model.eval().half()
    assert model


def test_loss():
    from tacotron2_model import Tacotron2Loss

    assert Tacotron2Loss()


def test_collate():
    from tacotron2_model import TextMelCollate

    assert TextMelCollate()


def test_stft():
    from tacotron2_model import TacotronSTFT

    assert TacotronSTFT()
