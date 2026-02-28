from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import tempfile


@dataclass
class TTSConfig:
    piper_bin: str = "piper"
    model_path: str = ""
    config_path: str = ""
    length_scale: float | None = None


class TTSService:
    def __init__(self, config: TTSConfig):
        self.cfg = config
        if not self.cfg.model_path or not os.path.isfile(self.cfg.model_path):
            raise FileNotFoundError(f"Modèle Piper introuvable: {self.cfg.model_path}")

    def synthesize(self, text: str) -> bytes:
        text = (text or "").strip()
        if not text:
            return b""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            cmd = [self.cfg.piper_bin, "--model", self.cfg.model_path, "--output_file", wav_path]

            if self.cfg.config_path:
                cmd += ["--config", self.cfg.config_path]
            if self.cfg.length_scale is not None:
                cmd += ["--length_scale", str(self.cfg.length_scale)]

            p = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            if p.returncode != 0:
                err = p.stderr.decode("utf-8", errors="ignore")
                raise RuntimeError(f"Piper TTS a échoué: {err}")

            with open(wav_path, "rb") as rf:
                return rf.read()

        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
