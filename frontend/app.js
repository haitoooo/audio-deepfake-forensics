let currentTab = "detect";
let selectedFile = null;
let selectedSample = "";
let traceChartInst = null;

function switchTab(tab) {
  currentTab = tab;
  ["detect", "sasv", "trace", "analyze"].forEach((t) => {
    const btn = document.getElementById(`tab-${t}`);
    btn.className =
      t === tab
        ? "tab-active py-3 px-4 rounded-xl text-sm font-semibold transition"
        : "glass py-3 px-4 rounded-xl text-sm font-semibold text-slate-300 hover:text-white transition";
  });

  document.getElementById("speakerPanel").classList.toggle("hidden", tab !== "sasv" && tab !== "analyze");
  document.getElementById("modelSelectWrap").classList.toggle("hidden", tab === "trace");
  document.getElementById("traceHeadWrap").classList.toggle("hidden", tab !== "trace" && tab !== "analyze");
  document.getElementById("alwaysTraceWrap").classList.toggle("hidden", tab !== "analyze");
}

async function initData() {
  const badge = document.getElementById("gpuBadge");
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    badge.textContent = `● 服务就绪 · ${h.gpu_name}`;
    badge.className = "text-xs px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-700 text-emerald-300";
  } catch (e) {
    badge.textContent = "● 服务连接异常";
    badge.className = "text-xs px-3 py-1 rounded-full bg-rose-950/80 border border-rose-700 text-rose-300";
    return;
  }

  const sData = await fetch("/api/samples").then((r) => r.json());
  const sel = document.getElementById("sampleSelect");
  sel.innerHTML = `<option value="">-- 选择示例语音 --</option>`;
  (sData.samples || []).forEach((s) => {
    const opt = document.createElement("option");
    opt.value = s.filename;
    opt.textContent = `${s.title} (${s.duration_sec}s)`;
    sel.appendChild(opt);
  });

  await refreshSpeakers();
}

async function refreshSpeakers() {
  const data = await fetch("/api/speakers").then((r) => r.json());
  const sel = document.getElementById("targetSpeakerSelect");
  const list = data.speakers || [];
  if (list.length === 0) {
    sel.innerHTML = `<option value="">-- 请先注册或一键导入示例人 --</option>`;
  } else {
    sel.innerHTML = list
      .map((s) => `<option value="${s.speaker_name}">${s.speaker_name}（已注册 ${s.num_utterances} 条语音）</option>`)
      .join("");
  }
}

async function quickEnrollDemoSpeaker() {
  const fd = new FormData();
  fd.append("speaker_name", "LA_0015");
  fd.append("sample_filenames", "target_LA0015_enroll_1.wav,target_LA0015_enroll_2.wav");
  fd.append("overwrite", "true");
  const res = await fetch("/api/speakers/enroll", { method: "POST", body: fd }).then((r) => r.json());
  if (res.detail) { alert("注册失败: " + res.detail); return; }
  await refreshSpeakers();
  document.getElementById("targetSpeakerSelect").value = "LA_0015";
  alert("已导入示例目标说话人 LA_0015（2 条真实语音）。现在可以上传或选择音频开始特定人防伪验证。");
}

async function enrollCurrentAudio() {
  const spk = document.getElementById("enrollSpeakerName").value.trim();
  if (!spk) { alert("请先输入说话人姓名或编号。"); return; }
  if (!selectedFile && !selectedSample) { alert("请先上传或选择一条目标人的真实语音。"); return; }
  const fd = new FormData();
  fd.append("speaker_name", spk);
  if (selectedFile) fd.append("files", selectedFile);
  else fd.append("sample_filenames", selectedSample);
  const res = await fetch("/api/speakers/enroll", { method: "POST", body: fd }).then((r) => r.json());
  if (res.detail) { alert("注册失败: " + res.detail); return; }
  await refreshSpeakers();
  document.getElementById("targetSpeakerSelect").value = spk;
  alert(`目标说话人「${spk}」注册成功（累计 ${res.num_utterances} 条语音）。`);
}

async function deleteSelectedSpeaker() {
  const spk = document.getElementById("targetSpeakerSelect").value;
  if (!spk) return;
  await fetch(`/api/speakers/${encodeURIComponent(spk)}`, { method: "DELETE" });
  await refreshSpeakers();
}

function renderRiskBar(spoofProb) {
  const wrap = document.getElementById("riskBarWrap");
  if (spoofProb === null || spoofProb === undefined) {
    wrap.classList.add("hidden");
    return;
  }
  wrap.classList.remove("hidden");
  const pct = Math.max(0, Math.min(100, Number(spoofProb)));
  const fill = document.getElementById("riskBarFill");
  const txt = document.getElementById("riskBarText");
  fill.style.width = `${pct}%`;
  if (pct >= 50) {
    fill.className = "h-full rounded-full bg-gradient-to-r from-amber-500 to-rose-500 transition-all duration-500";
    txt.className = "font-bold text-rose-400";
    txt.textContent = `${pct.toFixed(2)}%（高风险 · 疑似伪造）`;
  } else {
    fill.className = "h-full rounded-full bg-gradient-to-r from-emerald-500 to-cyan-400 transition-all duration-500";
    txt.className = "font-bold text-emerald-400";
    txt.textContent = `${pct.toFixed(2)}%（低风险 · 真实人声）`;
  }
}

function renderWaveform(peaks) {
  const box = document.getElementById("waveformBox");
  const bars = document.getElementById("waveformBars");
  if (!peaks || !peaks.length) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  bars.innerHTML = peaks
    .map((v) => `<div class="wave-bar flex-1 rounded-t bg-cyan-400/75" style="height:${Math.max(8, Math.round(v * 100))}%"></div>`)
    .join("");
}

function renderTraceChart(preds) {
  const box = document.getElementById("traceChartBox");
  if (!preds || !preds.length) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  const ctx = document.getElementById("traceChart").getContext("2d");
  if (traceChartInst) traceChartInst.destroy();
  traceChartInst = new Chart(ctx, {
    type: "bar",
    data: {
      labels: preds.map((p) => `${p.algorithm_id} (${p.category}) · ${p.description}`),
      datasets: [{
        label: "置信度 (%)",
        data: preds.map((p) => p.confidence_pct),
        backgroundColor: preds.map((_, i) => (i === 0 ? "#38bdf8" : "#475569")),
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: { max: 100, ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.1)" } },
        y: { ticks: { color: "#e2e8f0", font: { size: 11 } }, grid: { display: false } },
      },
    },
  });
}

function renderMetricCards(cards) {
  document.getElementById("metricGrid").innerHTML = cards
    .map((c) => `
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5">
        <div class="text-[11px] text-slate-400">${c.label}</div>
        <div class="text-base font-bold mt-1 ${c.color || "text-white"}">${c.value}</div>
      </div>`)
    .join("");
}

function setUploadedFile(file) {
  selectedFile = file;
  selectedSample = "";
  document.getElementById("sampleSelect").value = "";
  document.getElementById("selectedAudioLabel").textContent = file.name;
  const player = document.getElementById("audioPlayer");
  player.src = URL.createObjectURL(file);
  document.getElementById("audioPlayerWrap").classList.remove("hidden");
}

async function runInference() {
  if (!selectedFile && !selectedSample) {
    alert("请先上传一段语音文件，或从下拉框选择一条示例语音。");
    return;
  }

  const btn = document.getElementById("runBtn");
  btn.disabled = true;
  btn.textContent = "正在检测分析...";

  try {
    const fd = new FormData();
    if (selectedFile) fd.append("file", selectedFile);
    else fd.append("sample_filename", selectedSample);

    let endpoint = "/api/detect";
    if (currentTab === "detect") {
      fd.append("model", document.getElementById("cmModelSelect").value);
    } else if (currentTab === "sasv") {
      const spk = document.getElementById("targetSpeakerSelect").value;
      if (!spk) { alert("请先注册或点击「一键导入示例人 LA_0015」后再检测。"); return; }
      endpoint = "/api/verify-target";
      fd.append("speaker_name", spk);
      fd.append("cm_model", document.getElementById("cmModelSelect").value);
    } else if (currentTab === "trace") {
      endpoint = "/api/trace";
      fd.append("head", document.getElementById("traceHeadSelect").value);
      fd.append("top_k", "5");
    } else {
      endpoint = "/api/analyze";
      fd.append("model", document.getElementById("cmModelSelect").value);
      const spk = document.getElementById("targetSpeakerSelect").value;
      if (spk) fd.append("speaker_name", spk);
      fd.append("trace_head", document.getElementById("traceHeadSelect").value);
      fd.append("always_trace", document.getElementById("alwaysTraceCheck").checked ? "true" : "false");
    }

    const resp = await fetch(endpoint, { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) { alert("检测出错: " + (data.detail || resp.statusText)); return; }

    document.getElementById("emptyState").classList.add("hidden");
    document.getElementById("resultContent").classList.remove("hidden");

    const banner = document.getElementById("verdictBanner");
    const vTitle = document.getElementById("verdictTitle");
    const vSub = document.getElementById("verdictSub");
    const vLat = document.getElementById("verdictLatency");

    if (currentTab === "detect") {
      const isFake = data.verdict === "spoof";
      banner.className = isFake
        ? "rounded-xl p-5 border bg-rose-950/60 border-rose-700 text-rose-100 flex flex-wrap items-center justify-between gap-3"
        : "rounded-xl p-5 border bg-emerald-950/60 border-emerald-700 text-emerald-100 flex flex-wrap items-center justify-between gap-3";
      vSub.textContent = `非特定人伪造检测 · ${data.model}`;
      vTitle.textContent = data.verdict_zh;
      vLat.textContent = `${data.latency_ms} ms`;
      renderRiskBar(data.spoof_probability);
      renderMetricCards([
        { label: "伪造概率", value: `${data.spoof_probability}%`, color: isFake ? "text-rose-400" : "text-slate-200" },
        { label: "真声概率", value: `${data.bonafide_probability}%`, color: !isFake ? "text-emerald-400" : "text-slate-200" },
        { label: "音频时长", value: `${data.duration_sec} s` },
        { label: "采样率", value: `${data.orig_sample_rate} Hz` },
      ]);
      renderWaveform(data.waveform_preview);
      renderTraceChart(null);
    } else if (currentTab === "sasv") {
      const ok = data.accepted;
      banner.className = ok
        ? "rounded-xl p-5 border bg-emerald-950/60 border-emerald-700 text-emerald-100 flex flex-wrap items-center justify-between gap-3"
        : "rounded-xl p-5 border bg-rose-950/60 border-rose-700 text-rose-100 flex flex-wrap items-center justify-between gap-3";
      vSub.textContent = `特定人防伪验证 · 目标说话人：${data.speaker_name}`;
      vTitle.textContent = data.verdict_zh;
      vLat.textContent = `${data.latency_ms} ms`;
      renderRiskBar(data.cm_spoof_probability);
      renderMetricCards([
        { label: "声纹相似度", value: data.speaker_similarity, color: data.speaker_similarity >= 0.45 ? "text-emerald-400" : "text-amber-400" },
        { label: "防伪真声概率", value: `${data.cm_bonafide_probability}%`, color: data.cm_bonafide_probability >= 50 ? "text-emerald-400" : "text-rose-400" },
        { label: "综合防伪得分", value: data.sasv_fusion_score, color: "text-cyan-300" },
        { label: "核验结果", value: ok ? "通过" : "拦截拒绝", color: ok ? "text-emerald-400" : "text-rose-400" },
      ]);
      renderWaveform(data.waveform_preview);
      renderTraceChart(null);
    } else if (currentTab === "trace") {
      banner.className = "rounded-xl p-5 border bg-sky-950/60 border-sky-700 text-sky-100 flex flex-wrap items-center justify-between gap-3";
      vSub.textContent = "伪造音频算法溯源结果";
      vTitle.textContent = `${data.predicted_algorithm} · ${data.predicted_description}`;
      vLat.textContent = `${data.latency_ms} ms`;
      renderRiskBar(null);
      renderMetricCards([
        { label: "最可能算法", value: data.predicted_algorithm, color: "text-cyan-300" },
        { label: "技术类型", value: data.predicted_category },
        { label: "Top-1 置信度", value: `${data.top1_confidence_pct}%`, color: "text-emerald-400" },
        { label: "音频时长", value: `${data.duration_sec} s` },
      ]);
      renderWaveform(data.waveform_preview);
      renderTraceChart(data.top_predictions);
    } else {
      const det = data.detection;
      const sv = data.target_speaker_verification;
      const tr = data.algorithm_attribution;
      const isFake = det.verdict === "spoof";
      banner.className = isFake
        ? "rounded-xl p-5 border bg-rose-950/60 border-rose-700 text-rose-100 flex flex-wrap items-center justify-between gap-3"
        : "rounded-xl p-5 border bg-emerald-950/60 border-emerald-700 text-emerald-100 flex flex-wrap items-center justify-between gap-3";
      vSub.textContent = "一键综合检测结论";
      vTitle.textContent = sv ? sv.verdict_zh : det.verdict_zh;
      vLat.textContent = `${data.total_latency_ms} ms`;
      renderRiskBar(det.spoof_probability);
      renderMetricCards([
        { label: "伪造概率", value: `${det.spoof_probability}%`, color: isFake ? "text-rose-400" : "text-emerald-400" },
        { label: "声纹相似度", value: sv ? sv.speaker_similarity : "未选目标人", color: "text-cyan-300" },
        { label: "溯源算法", value: tr ? tr.predicted_algorithm : "真声无需溯源", color: "text-amber-300" },
        { label: "溯源置信度", value: tr ? `${tr.top1_confidence_pct}%` : "-" },
      ]);
      renderWaveform(det.waveform_preview);
      renderTraceChart(tr ? tr.top_predictions : null);
    }
  } finally {
    btn.disabled = false;
    btn.textContent = "开始检测";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("audioFileInput");
  const dropZone = document.getElementById("dropZone");
  const sampleSel = document.getElementById("sampleSelect");
  const player = document.getElementById("audioPlayer");
  const playerWrap = document.getElementById("audioPlayerWrap");
  const lbl = document.getElementById("selectedAudioLabel");

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) setUploadedFile(e.target.files[0]);
  });

  // 支持鼠标拖拽上传
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drop-active");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drop-active"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drop-active");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setUploadedFile(e.dataTransfer.files[0]);
  });

  sampleSel.addEventListener("change", (e) => {
    selectedSample = e.target.value;
    if (selectedSample) {
      selectedFile = null;
      fileInput.value = "";
      lbl.textContent = selectedSample;
      player.src = `/api/samples/${encodeURIComponent(selectedSample)}`;
      playerWrap.classList.remove("hidden");
    } else {
      lbl.textContent = "未选择音频";
      playerWrap.classList.add("hidden");
    }
  });

  initData();
});
