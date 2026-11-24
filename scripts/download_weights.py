"""一键下载大体积预训练权重（W2V2-AASIST 1.26GB 与 ECAPA-TDNN 64MB）。"""
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)

URLS = {
    "w2v2-aasist.onnx": "https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST/resolve/main/w2v2-aasist.onnx",
    "ecapa_pretrain.model": "https://github.com/TaoRuijie/ECAPA-TDNN/raw/main/exps/pretrain.model",
}


def download(name: str, url: str):
    dst = WEIGHTS / name
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        print(f"[SKIP] {dst.name} 已就绪 ({dst.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[DOWNLOAD] {name} <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dst, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        got = 0
        while True:
            buf = resp.read(1024 * 1024)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            if total:
                print(f"\r  {name}: {got/1e6:.1f}/{total/1e6:.1f} MB ({got/total*100:.1f}%)", end="", flush=True)
    print(f"\n[DONE] {dst}")


if __name__ == "__main__":
    for k, u in URLS.items():
        download(k, u)
