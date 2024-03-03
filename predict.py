# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from cog import BasePredictor, Input, Path
from melo.api import TTS

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        device = "cuda"
        self.models = {
            "EN": TTS(language="EN", device=device),
            "ES": TTS(language="ES", device=device),
            "FR": TTS(language="FR", device=device),
            "ZH": TTS(language="ZH", device=device),
            "JP": TTS(language="JP", device=device),
            "KR": TTS(language="KR", device=device),
        }

    def predict(
        self,
        language: str = Input(
            choices=["EN", "ES", "FR", "ZH", "JP", "KR"], default="EN"
        ),
        speaker: str = Input(
            description="For EN, choose a speaker, for other langauges, leave it blank.",
            choices=["EN-US", "EN-BR", "EN_INDIA", "EN-AU", "EN-Default", "-"],
            default="EN-US",
        ),
        text: str = Input(
            default="The field of text-to-speech has seen rapid development recently."
        ),
        speed: float = Input(
            description="Speed of the output.", default=1.0, ge=0.1, le=10.0
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        speaker_ids = self.models[language].hps.data.spk2id
        speaker_id = speaker_ids[speaker] if language == "EN" else speaker_ids[language]
        out_path = "/tmp/out.wav"
        self.models[language].tts_to_file(
            text, speaker_id, out_path, speed=speed, format="wav"
        )
        return Path(out_path)
