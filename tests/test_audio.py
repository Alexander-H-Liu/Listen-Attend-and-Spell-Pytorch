import unittest
import numpy as np
import torch

from src import audio


class TestAudio(unittest.TestCase):
    def setUp(self):
        super(TestAudio, self).__init__()
        self.filepath = "tests/sample_data/3830-12529-0005.wav"

    def test_filter_bank(self):
        audio_config = {
            "feat_type": "fbank",
            "feat_dim": 40,
            "apply_cmvn": False,
            "frame_length": 25,
            "frame_shift": 10,
        }

        transform, d = audio.create_transform(audio_config)
        y = transform(self.filepath)
        self.assertEqual(list(y.shape), [392, d])

    def test_mfcc(self):
        self.skipTest(
            "torchaudio.compliance.kaldi.mfcc is not in torchaudio==0.3.0")
        audio_config = {
            "feat_type": "mfcc",
            "feat_dim": 13,
            "apply_cmvn": False,
            "frame_length": 25,
            "frame_shift": 10,
        }

        transform, d = audio.create_transform(audio_config)
        y = transform(self.filepath)
        self.assertEqual(list(y.shape), [392, d])

    def test_cmvn(self):
        audio_config = {
            "feat_type": "fbank",
            "feat_dim": 40,
            "apply_cmvn": True,
            "frame_length": 25,
            "frame_shift": 10,
        }

        transform, d = audio.create_transform(audio_config)
        y = transform(self.filepath)

        self.assertEqual(list(y.shape), [392, d])
        np.testing.assert_allclose(y.mean(0), 0.0, rtol=1e-6, atol=5e-5)
        np.testing.assert_allclose(y.std(0), 1.0, rtol=1e-6, atol=1e-6)

    def test_delta(self):
        audio_config = {
            "feat_type": "fbank",
            "feat_dim": 40,
            "dither": 0.0,
            "apply_cmvn": True,
            "frame_length": 25,
            "frame_shift": 10,
            "delta_order": 1,
            "delta_window_size": 2,
        }

        transform, d = audio.create_transform(audio_config)
        y = transform(self.filepath)

        self.assertEqual(list(y.shape), [392, d])

        audio_config = {
            "feat_type": "fbank",
            "feat_dim": 40,
            "dither": 0.0,
            "apply_cmvn": True,
            "frame_length": 25,
            "frame_shift": 10,
            "delta_order": 0,
        }

        transform, d = audio.create_transform(audio_config)
        y_no_delta = transform(self.filepath)

        np.testing.assert_allclose(y[:, :40], y_no_delta, rtol=1e-5, atol=1e-5)

    def test_delta_delta(self):
        audio_config = {
            "feat_type": "fbank",
            "feat_dim": 40,
            "apply_cmvn": True,
            "frame_length": 25,
            "frame_shift": 10,
            "delta_order": 2,
            "delta_window_size": 2,
        }

        transform, d = audio.create_transform(audio_config)
        y = transform(self.filepath)

        self.assertEqual(list(y.shape), [392, d])
