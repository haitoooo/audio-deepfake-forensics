"""通用音频读取、16kHz 重采样、波形裁剪/填充与可视化缩略图工具。"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import soundfile as sf

TARGET_SR = 16000
MAX_LEN_SAMPLES = 64600  # ~4.04 秒 @ 16kHz（AASIST / W2V2-AASIST 标准输入长度）


def load_audio_16k(source: Union[str, Path, bytes]) -> Tuple[np.ndarray, int, float]:
    """加载音频文件或字节流，统一转为 16kHz 单声道 float32 波形。返回 (wav_16k, orig_sr, duration_sec)。"""
    if isinstance(source, (bytes, bytearray)):
        wav, sr = sf.read(io.BytesIO(source), dtype="float32")
    else:
        wav, sr = sf.read(str(source), dtype="float32")

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    orig_sr = int(sr)
    duration_sec = float(len(wav) / max(orig_sr, 1))

    if orig_sr != TARGET_SR and len(wav) > 1:
        new_len = max(1, int(round(len(wav) * TARGET_SR / orig_sr)))
        wav = np.interp(
            np.linspace(0, len(wav) - 1, new_len),
            np.arange(len(wav)),
            wav,
        ).astype(np.float32)

    return wav.astype(np.float32), orig_sr, round(duration_sec, 3)


def pad_or_trim(wav: np.ndarray, max_len: int = MAX_LEN_SAMPLES) -> np.ndarray:
    """按官方 AASIST / W2V2-AASIST 规范对波形进行循环填充或截断至固定长度。"""
    n = len(wav)
    if n == 0:
        return np.zeros(max_len, dtype=np.float32)
    if n >= max_len:
        return wav[:max_len].astype(np.float32)
    repeats = int(max_len / n) + 1
    return np.tile(wav, repeats)[:max_len].astype(np.float32)


def extract_waveform_preview(wav: np.ndarray, n_bins: int = 120) -> list[float]:
    """提取归一化包络预览序列，供前端轻量绘制波形。"""
    if len(wav) == 0:
        return [0.0] * n_bins
    chunks = np.array_split(np.abs(wav), n_bins)
    peaks = np.array([float(c.max()) if len(c) else 0.0 for c in chunks], dtype=np.float32)
    mx = float(peaks.max())
    if mx > 1e-6:
        peaks = peaks / mx
    return [round(float(v), 3) for v in peaks]
