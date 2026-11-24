"""
统一伪造音频检测、特定人防伪声纹验证（SASV）与合成算法溯源推理引擎。

支持三大核心能力：
  1. 非特定人伪造音频检测：W2V2-AASIST (317M SOTA) / AASIST (297K) / AASIST-L (85K)
  2. 特定人伪造音频检测：ECAPA-TDNN (192-d 声纹) + W2V2-AASIST/AASIST 防伪联合判别（SASV）
  3. 伪造音频算法溯源：端到端微调 AASIST-SourceTracer（13 类 A07-A19 与 6 类 A01-A06）
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio import extract_waveform_preview, load_audio_16k, pad_or_trim
from .models.aasist import Model as AASISTModel
from .models.ecapa_tdnn import ECAPA_TDNN

# 项目根目录默认自动定位（也支持通过环境变量 ASV_FORENSICS_ROOT 覆盖）
DEFAULT_ROOT = Path(os.environ.get("ASV_FORENSICS_ROOT", Path(__file__).resolve().parents[2]))

AASIST_CFG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

AASIST_L_CFG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

CLASSES_6 = ["A01", "A02", "A03", "A04", "A05", "A06"]
CLASSES_13 = ["A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19"]

ALGORITHM_INFO: Dict[str, Dict[str, str]] = {
    "A01": {"category": "TTS", "desc": "神经网络波形合成 (WaveNet)"},
    "A02": {"category": "TTS", "desc": "声学模型 + WORLD 声码器"},
    "A03": {"category": "TTS", "desc": "声学模型 + WORLD 声码器 (变体)"},
    "A04": {"category": "TTS", "desc": "波形拼接合成 (MaryTTS)"},
    "A05": {"category": "VC", "desc": "变分自编码器 VAE + WORLD 声码器"},
    "A06": {"category": "VC", "desc": "频谱滤波 + 重叠相加 (OLA)"},
    "A07": {"category": "TTS", "desc": "声码器 + 生成对抗网络 (GAN)"},
    "A08": {"category": "TTS", "desc": "神经自回归循环网络 (WaveRNN)"},
    "A09": {"category": "TTS", "desc": "声学模型 + WORLD 声码器"},
    "A10": {"category": "TTS", "desc": "端到端 Tacotron + WaveRNN"},
    "A11": {"category": "TTS", "desc": "端到端 Tacotron + Griffin-Lim"},
    "A12": {"category": "TTS", "desc": "深度自回归神经网络 (WaveNet)"},
    "A13": {"category": "TTS+VC", "desc": "波形拼接 + 波形滤波转换"},
    "A14": {"category": "TTS+VC", "desc": "声学模型 + STRAIGHT 声码器"},
    "A15": {"category": "TTS+VC", "desc": "神经声码器 WaveNet 联合转换"},
    "A16": {"category": "TTS", "desc": "波形拼接合成 (MaryTTS 变体)"},
    "A17": {"category": "VC", "desc": "变分自编码器 VAE + 波形滤波"},
    "A18": {"category": "VC", "desc": "非线性声码器 + MFCC/i-vector 转换"},
    "A19": {"category": "VC", "desc": "频谱滤波 + OLA 迁移转换"},
}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class ForensicsEngine:
    """多任务懒加载推理引擎（线程安全读取，支持 CUDA/CPU 自动切换）。"""

    def __init__(
        self,
        weights_dir: Optional[Union[str, Path]] = None,
        voiceprints_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.weights_dir = Path(weights_dir) if weights_dir else (DEFAULT_ROOT / "weights")
        self.voiceprints_dir = Path(voiceprints_dir) if voiceprints_dir else (DEFAULT_ROOT / "voiceprints")
        self.voiceprints_dir.mkdir(parents=True, exist_ok=True)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._aasist: Optional[AASISTModel] = None
        self._aasist_l: Optional[AASISTModel] = None
        self._w2v2_sess = None
        self._w2v2_gpu: bool = False
        self._ecapa: Optional[ECAPA_TDNN] = None
        self._tracer_13: Optional[AASISTModel] = None
        self._tracer_6: Optional[AASISTModel] = None

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        avail = {
            "w2v2_aasist": (self.weights_dir / "w2v2-aasist.onnx").exists(),
            "aasist": (self.weights_dir / "AASIST.pth").exists(),
            "aasist_l": (self.weights_dir / "AASIST-L.pth").exists(),
            "ecapa_tdnn": (self.weights_dir / "ecapa_pretrain.model").exists(),
            "tracer_13class": (self.weights_dir / "aasist_source_tracer_13class.pth").exists(),
            "tracer_6class": (self.weights_dir / "aasist_source_tracer_6class.pth").exists(),
        }
        return {
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
            "weights_dir": str(self.weights_dir),
            "available_models": avail,
            "enrolled_speakers": len(self.list_speakers()),
        }

    # ------------------------------------------------------------------
    # 懒加载模型
    # ------------------------------------------------------------------
    def _get_aasist(self) -> AASISTModel:
        if self._aasist is None:
            p = self.weights_dir / "AASIST.pth"
            if not p.exists():
                raise FileNotFoundError(f"缺少权重文件: {p}")
            m = AASISTModel(AASIST_CFG).to(self.device)
            m.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
            m.eval()
            self._aasist = m
        return self._aasist

    def _get_aasist_l(self) -> AASISTModel:
        if self._aasist_l is None:
            p = self.weights_dir / "AASIST-L.pth"
            if not p.exists():
                raise FileNotFoundError(f"缺少权重文件: {p}")
            m = AASISTModel(AASIST_L_CFG).to(self.device)
            m.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
            m.eval()
            self._aasist_l = m
        return self._aasist_l

    def _get_w2v2(self):
        if self._w2v2_sess is None:
            import onnxruntime as ort
            ort.set_default_logger_severity(3)

            p = self.weights_dir / "w2v2-aasist.onnx"
            if not p.exists():
                raise FileNotFoundError(
                    f"缺少 W2V2-AASIST 权重文件: {p}。请先运行 `python scripts/download_weights.py` 或改用 `--model aasist`。"
                )
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.intra_op_num_threads = 8
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
            sess = ort.InferenceSession(str(p), sess_options=so, providers=providers)
            self._w2v2_gpu = "CUDAExecutionProvider" in sess.get_providers()
            self._w2v2_sess = sess
        return self._w2v2_sess

    def _get_ecapa(self) -> ECAPA_TDNN:
        if self._ecapa is None:
            p = self.weights_dir / "ecapa_pretrain.model"
            if not p.exists():
                raise FileNotFoundError(f"缺少 ECAPA-TDNN 声纹权重文件: {p}。请先运行 `python scripts/download_weights.py`。")
            m = ECAPA_TDNN(C=1024).to(self.device)
            raw_sd = torch.load(p, map_location=self.device, weights_only=False)
            clean_sd = {}
            for k, v in raw_sd.items():
                if k.startswith("speaker_encoder."):
                    clean_sd[k[len("speaker_encoder."):]] = v
            m.load_state_dict(clean_sd, strict=True)
            m.eval()
            self._ecapa = m
        return self._ecapa

    def _get_tracer(self, head: str = "13class") -> AASISTModel:
        if head == "13class":
            if self._tracer_13 is None:
                p = self.weights_dir / "aasist_source_tracer_13class.pth"
                if not p.exists():
                    raise FileNotFoundError(f"缺少 13 类溯源权重: {p}")
                m = AASISTModel(AASIST_CFG)
                m.out_layer = nn.Linear(m.out_layer.in_features, len(CLASSES_13))
                m.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
                m = m.to(self.device).eval()
                self._tracer_13 = m
            return self._tracer_13
        elif head == "6class":
            if self._tracer_6 is None:
                p = self.weights_dir / "aasist_source_tracer_6class.pth"
                if not p.exists():
                    raise FileNotFoundError(f"缺少 6 类溯源权重: {p}")
                m = AASISTModel(AASIST_CFG)
                m.out_layer = nn.Linear(m.out_layer.in_features, len(CLASSES_6))
                m.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
                m = m.to(self.device).eval()
                self._tracer_6 = m
            return self._tracer_6
        raise ValueError(f"不支持的溯源模型头: {head}，可选 13class / 6class")

    # ------------------------------------------------------------------
    # 任务二：非特定人伪造音频检测
    # ------------------------------------------------------------------
    def detect(
        self,
        audio_source: Union[str, Path, bytes],
        model_name: str = "w2v2_aasist",
        threshold: Optional[float] = None,
        include_waveform: bool = False,
    ) -> Dict:
        wav, orig_sr, duration_sec = load_audio_16k(audio_source)
        wav_fixed = pad_or_trim(wav, 64600)

        t0 = time.perf_counter()
        if model_name == "w2v2_aasist":
            sess = self._get_w2v2()
            inp_name = sess.get_inputs()[0].name
            logits = sess.run(None, {inp_name: wav_fixed[None, :]})[0][0]
            # W2V2-AASIST: logits[0]=spoof, logits[1]=bonafide
            bonafide_score = float(logits[1])
            spoof_score = float(logits[0])
            thr = 0.0 if threshold is None else float(threshold)
            backend = "RTX GPU (ONNX-CUDA)" if self._w2v2_gpu else "CPU (ONNX)"
        elif model_name in ("aasist", "aasist_l"):
            m = self._get_aasist() if model_name == "aasist" else self._get_aasist_l()
            x = torch.from_numpy(wav_fixed[None, :]).to(self.device)
            with torch.no_grad():
                _, out = m(x)
            logits = out[0].cpu().numpy()
            bonafide_score = float(logits[1])
            spoof_score = float(logits[0])
            thr = 0.0 if threshold is None else float(threshold)
            backend = f"PyTorch ({self.device})"
        else:
            raise ValueError(f"不支持的检测模型: {model_name}，可选 w2v2_aasist / aasist / aasist_l")

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # softmax 概率（以相对阈值校准）
        delta = bonafide_score - spoof_score
        bonafide_prob = _sigmoid(delta)
        spoof_prob = 1.0 - bonafide_prob
        is_spoof = bonafide_score < thr

        res = {
            "task": "speaker_independent_detection",
            "model": model_name,
            "backend": backend,
            "verdict": "spoof" if is_spoof else "bonafide",
            "verdict_zh": "伪造音频 (Spoof)" if is_spoof else "真实人声 (Bonafide)",
            "spoof_probability": round(spoof_prob * 100.0, 2),
            "bonafide_probability": round(bonafide_prob * 100.0, 2),
            "bonafide_logit": round(bonafide_score, 4),
            "spoof_logit": round(spoof_score, 4),
            "threshold": round(thr, 4),
            "duration_sec": duration_sec,
            "orig_sample_rate": orig_sr,
            "latency_ms": latency_ms,
        }
        if include_waveform:
            res["waveform_preview"] = extract_waveform_preview(wav)
        return res

    # ------------------------------------------------------------------
    # 任务一：特定人声纹注册与特定人防伪检测（ECAPA-TDNN + AASIST/W2V2-AASIST SASV）
    # ------------------------------------------------------------------
    def _extract_speaker_embedding(self, wav_16k: np.ndarray) -> np.ndarray:
        ecapa = self._get_ecapa()
        x = torch.from_numpy(wav_16k[None, :]).float().to(self.device)
        with torch.no_grad():
            emb = ecapa.forward(x, aug=False)
            emb = F.normalize(emb, p=2, dim=1)
        return emb[0].cpu().numpy().astype(np.float32)

    def _speaker_path(self, speaker_name: str) -> Path:
        safe = "".join(c for c in speaker_name.strip() if c.isalnum() or c in ("_", "-", "."))
        if not safe:
            raise ValueError("说话人名称不能为空，仅支持字母、数字、下划线或连字符。")
        return self.voiceprints_dir / f"{safe}.json"

    def list_speakers(self) -> List[Dict]:
        items = []
        for p in sorted(self.voiceprints_dir.glob("*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                items.append({
                    "speaker_name": d["speaker_name"],
                    "num_utterances": d["num_utterances"],
                    "updated_at": d.get("updated_at", ""),
                })
            except Exception:
                continue
        return items

    def delete_speaker(self, speaker_name: str) -> bool:
        p = self._speaker_path(speaker_name)
        if p.exists():
            p.unlink()
            return True
        return False

    def enroll_speaker(
        self,
        speaker_name: str,
        audio_sources: List[Union[str, Path, bytes]],
        overwrite: bool = False,
    ) -> Dict:
        if not audio_sources:
            raise ValueError("请至少提供 1 条目标说话人的真实语音用于注册。")

        t0 = time.perf_counter()
        new_embs = []
        for src in audio_sources:
            wav, _, _ = load_audio_16k(src)
            new_embs.append(self._extract_speaker_embedding(wav))

        p = self._speaker_path(speaker_name)
        existing_embs = []
        if p.exists() and not overwrite:
            old = json.loads(p.read_text(encoding="utf-8"))
            existing_embs = [np.array(v, dtype=np.float32) for v in old.get("utterance_embeddings", [])]

        all_embs = existing_embs + new_embs
        centroid = np.mean(np.stack(all_embs, axis=0), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        record = {
            "speaker_name": speaker_name,
            "num_utterances": len(all_embs),
            "embedding_dim": int(len(centroid)),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "centroid": [float(v) for v in centroid],
            "utterance_embeddings": [[float(x) for x in e] for e in all_embs],
        }
        p.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

        return {
            "speaker_name": speaker_name,
            "num_utterances": len(all_embs),
            "embedding_dim": int(len(centroid)),
            "updated_at": record["updated_at"],
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def verify_target_speaker(
        self,
        speaker_name: str,
        audio_source: Union[str, Path, bytes],
        cm_model: str = "w2v2_aasist",
        sv_threshold: float = 0.45,
        cm_threshold: float = 0.0,
        sasv_threshold: float = 1.25,
        include_waveform: bool = False,
    ) -> Dict:
        """
        特定人防伪声纹验证（SASV）：
          - 通过 ECAPA-TDNN 计算待测音频与目标注册声纹的余弦相似度 (cos_sim ∈ [-1, 1])
          - 通过 W2V2-AASIST / AASIST 计算防伪真声概率 (cm_prob ∈ [0, 1])
          - 联合 SASV Baseline 1 Score-Sum 得分 = cos_sim + cm_prob，实现三态精细判别：
            1) target_bonafide: 目标人本人真实语音
            2) target_spoof:    针对该目标人的深度伪造攻击
            3) nontarget_impostor: 非目标人冒充（真人或非目标伪造）
        """
        p = self._speaker_path(speaker_name)
        if not p.exists():
            raise FileNotFoundError(f"未找到目标说话人 '{speaker_name}' 的注册声纹，请先执行声纹注册。")

        rec = json.loads(p.read_text(encoding="utf-8"))
        enr_emb = np.array(rec["centroid"], dtype=np.float32)

        t0 = time.perf_counter()
        wav, orig_sr, duration_sec = load_audio_16k(audio_source)
        tst_emb = self._extract_speaker_embedding(wav)
        cos_sim = float(np.dot(enr_emb, tst_emb) / (np.linalg.norm(enr_emb) * np.linalg.norm(tst_emb) + 1e-12))

        cm_res = self.detect(audio_source, model_name=cm_model, threshold=cm_threshold, include_waveform=False)
        cm_bonafide_prob = cm_res["bonafide_probability"] / 100.0
        sasv_score = cos_sim + cm_bonafide_prob

        is_spoof = cm_res["verdict"] == "spoof"
        is_same_speaker = cos_sim >= sv_threshold
        is_accepted = (not is_spoof) and is_same_speaker and (sasv_score >= sasv_threshold)

        if is_accepted:
            verdict = "target_bonafide"
            verdict_zh = f"验证通过：确认为目标人 [{speaker_name}] 真实人声"
        elif is_spoof and is_same_speaker:
            verdict = "target_spoof"
            verdict_zh = f"拦截告警：检测到针对目标人 [{speaker_name}] 的 AI 声音克隆/伪造攻击！"
        elif is_spoof:
            verdict = "spoof_attack"
            verdict_zh = "拦截告警：检测到伪造合成语音（且声纹不匹配目标人）"
        else:
            verdict = "nontarget_impostor"
            verdict_zh = f"验证拒绝：确认为真人语音，但非目标人 [{speaker_name}]（非目标人冒充）"

        res = {
            "task": "target_speaker_sasv",
            "speaker_name": speaker_name,
            "cm_model": cm_model,
            "sv_model": "ECAPA-TDNN (C=1024, 192-d)",
            "accepted": is_accepted,
            "verdict": verdict,
            "verdict_zh": verdict_zh,
            "speaker_similarity": round(cos_sim, 4),
            "speaker_similarity_pct": round(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)) * 100.0, 2),
            "cm_bonafide_probability": cm_res["bonafide_probability"],
            "cm_spoof_probability": cm_res["spoof_probability"],
            "sasv_fusion_score": round(sasv_score, 4),
            "thresholds": {
                "sv_threshold": sv_threshold,
                "cm_threshold": cm_threshold,
                "sasv_threshold": sasv_threshold,
            },
            "duration_sec": duration_sec,
            "orig_sample_rate": orig_sr,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
        if include_waveform:
            res["waveform_preview"] = extract_waveform_preview(wav)
        return res

    # ------------------------------------------------------------------
    # 任务三：伪造音频算法溯源（端到端微调 AASIST-SourceTracer）
    # ------------------------------------------------------------------
    def trace_algorithm(
        self,
        audio_source: Union[str, Path, bytes],
        head: str = "13class",
        top_k: int = 5,
        include_waveform: bool = False,
    ) -> Dict:
        wav, orig_sr, duration_sec = load_audio_16k(audio_source)
        wav_fixed = pad_or_trim(wav, 64600)
        classes = CLASSES_13 if head == "13class" else CLASSES_6

        t0 = time.perf_counter()
        model = self._get_tracer(head)
        x = torch.from_numpy(wav_fixed[None, :]).to(self.device)
        with torch.no_grad():
            _, logits = model(x)
            probs = F.softmax(logits[0], dim=-1).cpu().numpy()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        order = np.argsort(probs)[::-1]
        k = min(max(1, top_k), len(classes))
        top_predictions = []
        for rank, idx in enumerate(order[:k], start=1):
            alg_id = classes[int(idx)]
            info = ALGORITHM_INFO.get(alg_id, {"category": "Unknown", "desc": alg_id})
            top_predictions.append({
                "rank": rank,
                "algorithm_id": alg_id,
                "category": info["category"],
                "description": info["desc"],
                "confidence_pct": round(float(probs[int(idx)]) * 100.0, 2),
            })

        best = top_predictions[0]
        res = {
            "task": "algorithm_attribution",
            "model": f"AASIST-SourceTracer ({head})",
            "predicted_algorithm": best["algorithm_id"],
            "predicted_category": best["category"],
            "predicted_description": best["description"],
            "top1_confidence_pct": best["confidence_pct"],
            "top_predictions": top_predictions,
            "duration_sec": duration_sec,
            "orig_sample_rate": orig_sr,
            "latency_ms": latency_ms,
        }
        if include_waveform:
            res["waveform_preview"] = extract_waveform_preview(wav)
        return res

    # ------------------------------------------------------------------
    # 一键全流程综合取证分析
    # ------------------------------------------------------------------
    def full_analyze(
        self,
        audio_source: Union[str, Path, bytes],
        model_name: str = "w2v2_aasist",
        speaker_name: Optional[str] = None,
        trace_head: str = "13class",
        always_trace: bool = False,
        include_waveform: bool = True,
    ) -> Dict:
        t0 = time.perf_counter()
        det = self.detect(audio_source, model_name=model_name, include_waveform=include_waveform)

        sasv = None
        if speaker_name:
            sasv = self.verify_target_speaker(speaker_name, audio_source, cm_model=model_name, include_waveform=False)

        trace = None
        if det["verdict"] == "spoof" or always_trace:
            trace = self.trace_algorithm(audio_source, head=trace_head, top_k=5, include_waveform=False)

        return {
            "pipeline": "full_audio_deepfake_forensics",
            "detection": det,
            "target_speaker_verification": sasv,
            "algorithm_attribution": trace,
            "total_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
