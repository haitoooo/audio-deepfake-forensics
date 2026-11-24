"""通用音频读取、万能格式解码（支持 M4A/AAC/AMR/MP3/MP4/WebM 等）、16kHz 重采样与波形预览。"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import soundfile as sf

TARGET_SR = 16000
MAX_LEN_SAMPLES = 64600  # ~4.04 秒 @ 16kHz（AASIST / W2V2-AASIST 标准输入长度）


def _decode_with_ffmpeg(source: Union[str, Path, bytes]) -> Tuple[np.ndarray, int, float]:
    """使用 ffmpeg 万能解码任意手机录音/容器格式（m4a/aac/amr/mp3/mp4/webm/3gp/opus等）为 16kHz 单声道 float32。"""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("当前音频格式需要 ffmpeg 解码，但系统未安装 ffmpeg。请安装 ffmpeg 或转为 WAV/FLAC 后重试。")

    tmp_path = None
    try:
        if isinstance(source, (bytes, bytearray)):
            # 写入临时文件，确保 m4a/mp4（moov atom 可能在末尾）可正常 seek
            fd, tmp_path = tempfile.mkstemp(prefix="asv_audio_")
            with os.fdopen(fd, "wb") as f:
                f.write(source)
            in_path = tmp_path
        else:
            in_path = str(source)

        # 用 ffprobe 探测原始采样率（失败则默认 16000）
        orig_sr = TARGET_SR
        ffprobe_bin = shutil.which("ffprobe")
        if ffprobe_bin:
            probe = subprocess.run(
                [
                    ffprobe_bin,
                    "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=sample_rate",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    in_path,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if probe.returncode == 0 and probe.stdout.strip().isdigit():
                orig_sr = int(probe.stdout.strip())

        cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-v", "error",
            "-i", in_path,
            "-ac", "1",
            "-ar", str(TARGET_SR),
            "-f", "f32le",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
        if proc.returncode != 0 or len(proc.stdout) < 4:
            err_msg = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"音频解码失败（请确认文件未损坏且包含音轨）: {err_msg}")

        wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        duration_sec = float(len(wav) / TARGET_SR)
        return wav, orig_sr, round(duration_sec, 3)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def load_audio_16k(source: Union[str, Path, bytes]) -> Tuple[np.ndarray, int, float]:
    """加载任意格式音频文件或字节流，统一转为 16kHz 单声道 float32 波形。返回 (wav_16k, orig_sr, duration_sec)。"""
    try:
        if isinstance(source, (bytes, bytearray)):
            wav, sr = sf.read(io.BytesIO(source), dtype="float32")
        else:
            wav, sr = sf.read(str(source), dtype="float32")
    except Exception:
        # soundfile 不支持 M4A / AAC / AMR / MP4 / WebM 等手机录音格式，自动回退 ffmpeg 万能解码
        return _decode_with_ffmpeg(source)

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
