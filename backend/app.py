"""
Audio Deepfake Forensics FastAPI 推理后端服务。

提供标准 RESTful API（自动生成 /docs Swagger 交互文档）与前后端分离 CORS 支持，
同时在根路径 `/` 挂载前端静态页面，支持「独立前后端分离部署」或「单命令一体化启动」。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from asv_forensics.engine import DEFAULT_ROOT, ForensicsEngine

SAMPLES_DIR = DEFAULT_ROOT / "samples"
FRONTEND_DIR = DEFAULT_ROOT / "frontend"

app = FastAPI(
    title="Audio Deepfake Forensics API",
    description="伪造音频检测、特定人防伪声纹验证（SASV）与合成算法取证溯源统一后端服务",
    version="1.0.0",
)

# 允许前后端分离跨域调用（如独立前端端口 5173 / 3000 / 8080 访问后端 8000）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ForensicsEngine()


def _resolve_audio_bytes(file: Optional[UploadFile], sample_filename: Optional[str]) -> tuple[bytes, str]:
    if sample_filename:
        safe_name = Path(sample_filename).name
        p = SAMPLES_DIR / safe_name
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"示例音频不存在: {safe_name}")
        return p.read_bytes(), safe_name
    if file is not None:
        return file.file.read(), (file.filename or "uploaded.wav")
    raise HTTPException(status_code=400, detail="请上传音频文件或选择一个内置示例音频。")


@app.get("/api/health", tags=["System"])
def api_health():
    """查看后端设备状态（GPU/CPU）、已就绪模型权重及已注册说话人数量。"""
    return {"status": "ok", **engine.get_status()}


@app.get("/api/samples", tags=["Samples"])
def api_samples():
    """获取内置演示音频清单（源自 ASVspoof 2019 LA 评估集）。"""
    mf = SAMPLES_DIR / "manifest.json"
    if not mf.exists():
        return {"samples": []}
    return {"samples": json.loads(mf.read_text(encoding="utf-8"))}


@app.get("/api/samples/{filename}", tags=["Samples"])
def api_sample_file(filename: str):
    """下载或在线试听指定内置演示音频。"""
    p = SAMPLES_DIR / Path(filename).name
    if not p.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(p, media_type="audio/wav", filename=p.name)


@app.post("/api/detect", tags=["Task 2: Speaker-Independent Detection"])
async def api_detect(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    model: str = Form("w2v2_aasist"),
    threshold: Optional[float] = Form(None),
):
    """任务二：非特定人伪造音频检测（支持 W2V2-AASIST / AASIST / AASIST-L）。"""
    raw, fname = _resolve_audio_bytes(file, sample_filename)
    try:
        res = engine.detect(raw, model_name=model, threshold=threshold, include_waveform=True)
        res["filename"] = fname
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/speakers", tags=["Task 1: Target-Speaker SASV"])
def api_list_speakers():
    """列出声纹库中已注册的目标说话人。"""
    return {"speakers": engine.list_speakers()}


@app.post("/api/speakers/enroll", tags=["Task 1: Target-Speaker SASV"])
async def api_enroll_speaker(
    speaker_name: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    sample_filenames: Optional[str] = Form(None),
    overwrite: bool = Form(False),
):
    """任务一（步骤1）：上传 1 条或多条目标人真实语音（或选择内置示例）完成 ECAPA-TDNN 声纹注册。"""
    audio_list: List[bytes] = []
    if sample_filenames:
        for fn in [x.strip() for x in sample_filenames.split(",") if x.strip()]:
            p = SAMPLES_DIR / Path(fn).name
            if not p.exists():
                raise HTTPException(status_code=404, detail=f"示例音频不存在: {fn}")
            audio_list.append(p.read_bytes())
    if files:
        for f in files:
            data = await f.read()
            if data:
                audio_list.append(data)
    if not audio_list:
        raise HTTPException(status_code=400, detail="请至少提供 1 条注册音频或选择示例。")
    try:
        return engine.enroll_speaker(speaker_name, audio_list, overwrite=overwrite)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/speakers/{speaker_name}", tags=["Task 1: Target-Speaker SASV"])
def api_delete_speaker(speaker_name: str):
    """从声纹库删除指定目标说话人。"""
    ok = engine.delete_speaker(speaker_name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"未找到说话人: {speaker_name}")
    return {"deleted": speaker_name}


@app.post("/api/verify-target", tags=["Task 1: Target-Speaker SASV"])
async def api_verify_target(
    speaker_name: str = Form(...),
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    cm_model: str = Form("w2v2_aasist"),
    sv_threshold: float = Form(0.45),
    sasv_threshold: float = Form(1.25),
):
    """任务一（步骤2）：特定人防伪声纹验证（ECAPA-TDNN + W2V2-AASIST/AASIST 联合判别）。"""
    raw, fname = _resolve_audio_bytes(file, sample_filename)
    try:
        res = engine.verify_target_speaker(
            speaker_name=speaker_name,
            audio_source=raw,
            cm_model=cm_model,
            sv_threshold=sv_threshold,
            sasv_threshold=sasv_threshold,
            include_waveform=True,
        )
        res["filename"] = fname
        return res
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/trace", tags=["Task 3: Algorithm Attribution"])
async def api_trace(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    head: str = Form("13class"),
    top_k: int = Form(5),
):
    """任务三：伪造音频合成/变声算法取证溯源（13类 A07-A19 / 6类 A01-A06）。"""
    raw, fname = _resolve_audio_bytes(file, sample_filename)
    try:
        res = engine.trace_algorithm(raw, head=head, top_k=top_k, include_waveform=True)
        res["filename"] = fname
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze", tags=["Full Pipeline"])
async def api_analyze(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    model: str = Form("w2v2_aasist"),
    speaker_name: Optional[str] = Form(None),
    trace_head: str = Form("13class"),
    always_trace: bool = Form(False),
):
    """一键全流程综合取证分析（真伪检测 + 可选特定人验证 + 算法溯源）。"""
    raw, fname = _resolve_audio_bytes(file, sample_filename)
    try:
        res = engine.full_analyze(
            raw,
            model_name=model,
            speaker_name=speaker_name or None,
            trace_head=trace_head,
            always_trace=always_trace,
            include_waveform=True,
        )
        res["filename"] = fname
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 若存在 frontend 目录则挂载静态文件（方便单端口预览，同时保留独立前端分离部署能力）
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
