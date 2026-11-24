"""
Audio Deepfake Forensics CLI 命令行调用工具。

支持命令：
  asv-forensics detect        非特定人伪造音频检测（支持单文件或目录批量导出 CSV/JSON）
  asv-forensics enroll        目标说话人真实声纹注册
  asv-forensics verify-target 特定人防伪声纹验证（SASV）
  asv-forensics speakers      查看或删除已注册的目标说话人
  asv-forensics trace         伪造音频合成/变声算法溯源
  asv-forensics analyze       一键全流程综合取证分析
  asv-forensics status        查看硬件设备与已加载模型状态
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .engine import ForensicsEngine

app = typer.Typer(
    name="asv-forensics",
    help="伪造音频检测、特定人防伪声纹验证（SASV）与合成算法溯源 CLI 工具",
    no_args_is_help=True,
)
console = Console()

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _engine(weights_dir: Optional[str], device: Optional[str]) -> ForensicsEngine:
    return ForensicsEngine(weights_dir=weights_dir, device=device)


@app.command("status")
def cmd_status(
    weights_dir: Optional[str] = typer.Option(None, "--weights-dir", help="模型权重目录"),
    device: Optional[str] = typer.Option(None, "--device", help="推理设备 (cuda / cpu)"),
):
    """查看当前推理设备、可用模型权重与已注册说话人数量。"""
    st = _engine(weights_dir, device).get_status()
    tbl = Table(title="Audio Deepfake Forensics 系统状态", show_header=True)
    tbl.add_column("项目", style="cyan")
    tbl.add_column("状态", style="bold")
    tbl.add_row("计算设备", f"{st['gpu_name']} ({st['device']})")
    tbl.add_row("权重目录", st["weights_dir"])
    tbl.add_row("已注册目标说话人", str(st["enrolled_speakers"]))
    for m, ok in st["available_models"].items():
        tbl.add_row(f"模型 · {m}", "[green]就绪 ✓[/green]" if ok else "[yellow]未下载[/yellow]")
    console.print(tbl)


@app.command("detect")
def cmd_detect(
    target: Path = typer.Argument(..., exists=True, help="待检测的音频文件或目录"),
    model: str = typer.Option("w2v2_aasist", "--model", "-m", help="检测模型: w2v2_aasist | aasist | aasist_l"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="自定义 Bonafide 判决阈值"),
    output_json: Optional[Path] = typer.Option(None, "--json", help="保存 JSON 结果路径"),
    output_csv: Optional[Path] = typer.Option(None, "--csv", help="批量检测时保存 CSV 路径"),
    weights_dir: Optional[str] = typer.Option(None, "--weights-dir"),
    device: Optional[str] = typer.Option(None, "--device"),
):
    """任务二：非特定人伪造音频检测（支持单文件或目录批量检测）。"""
    eng = _engine(weights_dir, device)
    files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        console.print("[red]未在指定目录中找到音频文件。[/red]")
        raise typer.Exit(1)

    results = []
    tbl = Table(title=f"非特定人伪造音频检测结果 (模型: {model})", show_header=True)
    tbl.add_column("音频文件", style="cyan")
    tbl.add_column("判定结果", justify="center")
    tbl.add_column("伪造概率", justify="right")
    tbl.add_column("真声 Logit", justify="right")
    tbl.add_column("时长", justify="right")
    tbl.add_column("耗时", justify="right")

    for f in files:
        r = eng.detect(f, model_name=model, threshold=threshold)
        r["file"] = str(f)
        results.append(r)
        color = "bold red" if r["verdict"] == "spoof" else "bold green"
        tbl.add_row(
            f.name,
            f"[{color}]{r['verdict_zh']}[/{color}]",
            f"{r['spoof_probability']:.2f}%",
            f"{r['bonafide_logit']:+.4f}",
            f"{r['duration_sec']:.2f}s",
            f"{r['latency_ms']:.1f}ms",
        )

    console.print(tbl)

    if output_json:
        output_json.write_text(json.dumps(results if len(results) > 1 else results[0], indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]JSON 结果已保存至: {output_json}[/green]")
    if output_csv:
        keys = ["file", "model", "verdict", "spoof_probability", "bonafide_probability", "bonafide_logit", "duration_sec", "latency_ms"]
        with open(output_csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in keys})
        console.print(f"[green]CSV 结果已保存至: {output_csv}[/green]")


@app.command("enroll")
def cmd_enroll(
    speaker: str = typer.Option(..., "--speaker", "-s", help="目标说话人 ID/姓名"),
    audios: List[Path] = typer.Argument(..., exists=True, help="1 条或多条目标说话人真实语音"),
    overwrite: bool = typer.Option(False, "--overwrite", help="是否覆盖已有注册声纹"),
    weights_dir: Optional[str] = typer.Option(None, "--weights-dir"),
    device: Optional[str] = typer.Option(None, "--device"),
):
    """任务一（步骤1）：提取 ECAPA-TDNN 声纹并注册目标说话人。"""
    eng = _engine(weights_dir, device)
    res = eng.enroll_speaker(speaker, audios, overwrite=overwrite)
    console.print(
        Panel.fit(
            f"[bold green]目标说话人声纹注册成功[/bold green]\n"
            f"说话人: [cyan]{res['speaker_name']}[/cyan]\n"
            f"累计注册音频数: {res['num_utterances']} 条\n"
            f"声纹维度: {res['embedding_dim']}-d (ECAPA-TDNN)\n"
            f"处理耗时: {res['latency_ms']:.1f} ms",
            title="特定人声纹库",
        )
    )


@app.command("verify-target")
def cmd_verify_target(
    speaker: str = typer.Option(..., "--speaker", "-s", help="已注册的目标说话人 ID"),
    audio: Path = typer.Argument(..., exists=True, help="待验证的音频文件"),
    cm_model: str = typer.Option("w2v2_aasist", "--model", "-m", help="防伪后端模型: w2v2_aasist | aasist | aasist_l"),
    sv_threshold: float = typer.Option(0.45, "--sv-threshold", help="ECAPA 声纹相似度阈值"),
    sasv_threshold: float = typer.Option(1.25, "--sasv-threshold", help="SASV 融合得分阈值"),
    output_json: Optional[Path] = typer.Option(None, "--json", help="保存 JSON 结果路径"),
    weights_dir: Optional[str] = typer.Option(None, "--weights-dir"),
    device: Optional[str] = typer.Option(None, "--device"),
):
    """任务一（步骤2）：特定人伪造音频检测（ECAPA-TDNN 声纹 + 防伪模型 SASV 联合判别）。"""
    eng = _engine(weights_dir, device)
    r = eng.verify_target_speaker(
        speaker_name=speaker,
        audio_source=audio,
        cm_model=cm_model,
        sv_threshold=sv_threshold,
        sasv_threshold=sasv_threshold,
    )
    style = "green" if r["accepted"] else "red"
    console.print(
        Panel.fit(
            f"[bold {style}]{r['verdict_zh']}[/bold {style}]\n\n"
            f"目标说话人: [cyan]{r['speaker_name']}[/cyan] | 测试音频: {audio.name}\n"
            f"声纹余弦相似度 (ECAPA-TDNN): [bold]{r['speaker_similarity']:.4f}[/bold] (阈值 ≥ {sv_threshold})\n"
            f"防伪真声概率 ({cm_model}): [bold]{r['cm_bonafide_probability']:.2f}%[/bold] (伪造概率 {r['cm_spoof_probability']:.2f}%)\n"
            f"SASV 融合得分 (Score-Sum): [bold]{r['sasv_fusion_score']:.4f}[/bold] (阈值 ≥ {sasv_threshold})\n"
            f"总推理耗时: {r['latency_ms']:.1f} ms",
            title="特定人伪造音频检测 (SASV)",
            border_style=style,
        )
    )
    if output_json:
        output_json.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")


@app.command("speakers")
def cmd_speakers(
    delete: Optional[str] = typer.Option(None, "--delete", help="删除指定说话人声纹"),
    weights_dir: Optional[str] = typer.Option(None, "--weights-dir"),
):
    """查看或管理已注册的特定人声纹库。"""
    eng = _engine(weights_dir, "cpu")
    if delete:
        ok = eng.delete_speaker(delete)
        console.print(f"[green]已删除说话人 {delete}[/green]" if ok else f"[yellow]未找到说话人 {delete}[/yellow]")
        return
    spks = eng.list_speakers()
    tbl = Table(title="已注册特定人声纹库")
    tbl.add_column("说话人 ID", style="cyan")
    tbl.add_column("注册音频数", justify="right")
    tbl.add_column("更新时间")
    for s in spks:
        tbl.add_row(s["speaker_name"], str(s["num_utterances"]), s["updated_at"])
    console.print(tbl if spks else "[yellow]当前声纹库为空，请先使用 `asv-forensics enroll` 注册目标人。[/yellow]")


@app.command("trace")
def cmd_trace(
    audio: Path = typer.Argument(..., exists=True, help="待溯源的伪造音频文件"),
    head: str = typer.Option("13class", "--head", "-h", help="溯源分类头: 13class (A07-A19) | 6class (A01-A06)"),
    top_k: int = typer.Option(5, "--topk", "-k", help="展示 Top-K 疑似合成算法"),
    output_json: Optional[Path] = typer.Option(None, "--json", help="保存 JSON 结果路径"),
    weights_dir: Optional[str] = typer.Option(None, "--weights-dir"),
    device: Optional[str] = typer.Option(None, "--device"),
):
    """任务三：伪造音频合成/变声算法取证溯源。"""
    eng = _engine(weights_dir, device)
    r = eng.trace_algorithm(audio, head=head, top_k=top_k)

    tbl = Table(title=f"伪造音频算法溯源结果 ({audio.name} · {r['model']} · {r['latency_ms']:.1f}ms)")
    tbl.add_column("排名", justify="center", style="bold")
    tbl.add_column("算法编号", style="cyan bold")
    tbl.add_column("类型", justify="center")
    tbl.add_column("合成/变声技术路线")
    tbl.add_column("置信度", justify="right", style="bold green")
    for item in r["top_predictions"]:
        tbl.add_row(
            f"#{item['rank']}",
            item["algorithm_id"],
            item["category"],
            item["description"],
            f"{item['confidence_pct']:.2f}%",
        )
    console.print(tbl)
    if output_json:
        output_json.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")


@app.command("analyze")
def cmd_analyze(
    audio: Path = typer.Argument(..., exists=True, help="待综合分析的音频文件"),
    model: str = typer.Option("w2v2_aasist", "--model", "-m", help="防伪检测模型"),
    speaker: Optional[str] = typer.Option(None, "--speaker", "-s", help="可选：指定已注册目标人进行 SASV 特定人验证"),
    head: str = typer.Option("13class", "--head", help="算法溯源头: 13class | 6class"),
    always_trace: bool = typer.Option(False, "--always-trace", help="即使判定为真声也强制输出算法溯源分布"),
    output_json: Optional[Path] = typer.Option(None, "--json", help="保存完整综合取证 JSON"),
    weights_dir: Optional[str] = typer.Option(None, "--weights-dir"),
    device: Optional[str] = typer.Option(None, "--device"),
):
    """一键全流程综合取证（自动串联真伪检测 + 特定人验证 + 算法溯源）。"""
    eng = _engine(weights_dir, device)
    r = eng.full_analyze(audio, model_name=model, speaker_name=speaker, trace_head=head, always_trace=always_trace, include_waveform=False)

    det = r["detection"]
    det_color = "red" if det["verdict"] == "spoof" else "green"
    lines = [
        f"[bold]1. 非特定人真伪检测 ({model}):[/bold] [{det_color}]{det['verdict_zh']}[/{det_color}] "
        f"(伪造概率 {det['spoof_probability']:.2f}%, 耗时 {det['latency_ms']:.1f}ms)"
    ]
    if r["target_speaker_verification"]:
        sv = r["target_speaker_verification"]
        sv_c = "green" if sv["accepted"] else "red"
        lines.append(
            f"[bold]2. 特定人声纹防伪 ({speaker}):[/bold] [{sv_c}]{sv['verdict_zh']}[/{sv_c}] "
            f"(相似度 {sv['speaker_similarity']:.4f}, SASV得分 {sv['sasv_fusion_score']:.4f})"
        )
    if r["algorithm_attribution"]:
        tr = r["algorithm_attribution"]
        lines.append(
            f"[bold]3. 伪造算法溯源 ({head}):[/bold] [cyan]{tr['predicted_algorithm']} · {tr['predicted_description']}[/cyan] "
            f"(Top-1 置信度 {tr['top1_confidence_pct']:.2f}%)"
        )
    lines.append(f"\n[dim]全流程总耗时: {r['total_latency_ms']:.1f} ms[/dim]")
    console.print(Panel.fit("\n".join(lines), title=f"综合取证报告 · {audio.name}"))

    if output_json:
        output_json.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    app()
