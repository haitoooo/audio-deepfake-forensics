let currentTab = "detect";
let selectedFile = null;
let selectedSample = "";
let traceChartInst = null;

function getApiBase() {
  const v = (document.getElementById("apiBaseInput").value || "").trim().replace(/\/+$/, "");
  return v || window.location.origin;
}

function saveApiBase() {
  localStorage.setItem("asv_api_base", document.getElementById("apiBaseInput").value.trim());
  document.getElementById("swaggerLink").href = `${getApiBase()}/docs`;
  initData();
}

function switchTab(tab) {
  currentTab = tab;
  ["detect", "sasv", "trace", "analyze"].forEach((t) => {
    const btn = document.getElementById(`tab-${t}`);
    if (t === tab) {
      btn.className = "tab-active py-3 px-4 rounded-xl text-sm font-semibold transition";
    } else {
      btn.className = "glass py-3 px-4 rounded-xl text-sm font-semibold text-slate-300 hover:text-white transition";
    }
  });

  document.getElementById("speakerPanel").classList.toggle("hidden", tab !== "sasv" && tab !== "analyze");
  document.getElementById("modelSelectWrap").classList.toggle("hidden", tab === "trace");
  document.getElementById("traceHeadWrap").classList.toggle("hidden", tab !== "trace" && tab !== "analyze");
  document.getElementById("alwaysTraceWrap").classList.toggle("hidden", tab !== "analyze");
  updateSnippets();
}

function updateSnippets() {
  const audioArg = selectedSample ? `samples/${selectedSample}` : (selectedFile ? selectedFile.name : "samples/target_LA0015_spoof_A10.wav");
  const model = document.getElementById("cmModelSelect").value;
  const head = document.getElementById("traceHeadSelect").value;
  const spk = document.getElementById("targetSpeakerSelect").value || "LA_0015";
  const api = getApiBase();

  let cli = "", curl = "";
  if (currentTab === "detect") {
    cli = `asv-forensics detect ${audioArg} --model ${model}`;
    curl = `curl -X POST ${api}/api/detect -F "file=@${audioArg}" -F "model=${model}"`;
  } else if (currentTab === "sasv") {
    cli = `asv-forensics verify-target --speaker ${spk} ${audioArg} --model ${model}`;
    curl = `curl -X POST ${api}/api/verify-target -F "speaker_name=${spk}" -F "file=@${audioArg}" -F "cm_model=${model}"`;
  } else if (currentTab === "trace") {
    cli = `asv-forensics trace ${audioArg} --head ${head} --topk 5`;
    curl = `curl -X POST ${api}/api/trace -F "file=@${audioArg}" -F "head=${head}" -F "top_k=5"`;
  } else {
    cli = `asv-forensics analyze ${audioArg} --model ${model} --speaker ${spk} --head ${head}`;
    curl = `curl -X POST ${api}/api/analyze -F "file=@${audioArg}" -F "model=${model}" -F "speaker_name=${spk}" -F "trace_head=${head}"`;
  }
  document.getElementById("cliSnippet").textContent = `$ ${cli}`;
  document.getElementById("curlSnippet").textContent = `$ ${curl}`;
}

async function initData() {
  const api = getApiBase();
  const badge = document.getElementById("gpuBadge");
  try {
    const h = await fetch(`${api}/api/health`).then((r) => r.json());
    badge.textContent = `● 在线 · ${h.gpu_name}`;
    badge.className = "px-2.5 py-1 rounded-full bg-emerald-950/80 border border-emerald-700 text-emerald-300";
  } catch (e) {
    badge.textContent = "● 后端未连接";
    badge.className = "px-2.5 py-1 rounded-full bg-rose-950/80 border border-rose-700 text-rose-300";
    return;
  }

  // 加载示例音频
  const sData = await fetch(`${api}/api/samples`).then((r) => r.json());
  const sel = document.getElementById("sampleSelect");
  sel.innerHTML = `<option value="">-- 请选择内置真实/伪造示例音频 --</option>`;
  (sData.samples || []).forEach((s) => {
    const opt = document.createElement("option");
    opt.value = s.filename;
    opt.textContent = `${s.title} (${s.duration_sec}s)`;
    sel.appendChild(opt);
  });

  await refreshSpeakers();
  updateSnippets();
}

async function refreshSpeakers() {
  const api = getApiBase();
  const data = await fetch(`${api}/api/speakers`).then((r) => r.json());
  const sel = document.getElementById("targetSpeakerSelect");
  const list = data.speakers || [];
  if (list.length === 0) {
    sel.innerHTML = `<option value="">-- 暂无已注册说话人（请点击上方一键注册） --</option>`;
  } else {
    sel.innerHTML = list
      .map((s) => `<option value="${s.speaker_name}">${s.speaker_name} (${s.num_utterances} 条注册语音)</option>`)
      .join("");
  }
  updateSnippets();
}

async function quickEnrollDemoSpeaker() {
  const api = getApiBase();
  const fd = new FormData();
  fd.append("speaker_name", "LA_0015");
  fd.append("sample_filenames", "target_LA0015_enroll_1.wav,target_LA0015_enroll_2.wav");
  fd.append("overwrite", "true");
  const res = await fetch(`${api}/api/speakers/enroll`, { method: "POST", body: fd }).then((r) => r.json());
  if (res.detail) { alert("注册失败: " + res.detail); return; }
  await refreshSpeakers();
  document.getElementById("targetSpeakerSelect").value = "LA_0015";
  alert(`已成功用 2 条真实语音注册示例目标人 LA_0015（耗时 ${res.latency_ms} ms）！现在可选取 LA_0015 的真声、伪造攻击或冒充音频进行测试。`);
}

async function enrollCurrentAudio() {
  const spk = document.getElementById("enrollSpeakerName").value.trim();
  if (!spk) { alert("请先输入目标说话人 ID！"); return; }
  if (!selectedFile && !selectedSample) { alert("请先在上方上传或选择 1 条真实语音！"); return; }
  const api = getApiBase();
  const fd = new FormData();
  fd.append("speaker_name", spk);
  if (selectedFile) fd.append("files", selectedFile);
  else fd.append("sample_filenames", selectedSample);
  const res = await fetch(`${api}/api/speakers/enroll`, { method: "POST", body: fd }).then((r) => r.json());
  if (res.detail) { alert("注册失败: " + res.detail); return; }
  await refreshSpeakers();
  document.getElementById("targetSpeakerSelect").value = spk;
  alert(`目标说话人 [${spk}] 注册成功（累计 ${res.num_utterances} 条，耗时 ${res.latency_ms} ms）`);
}

async function deleteSelectedSpeaker() {
  const spk = document.getElementById("targetSpeakerSelect").value;
  if (!spk) return;
  await fetch(`${getApiBase()}/api/speakers/${encodeURIComponent(spk)}`, { method: "DELETE" });
  await refreshSpeakers();
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
        backgroundColor: preds.map((_, i) => i === 0 ? "#38bdf8" : "#475569"),
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
      <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-3">
        <div class="text-[11px] text-slate-400">${c.label}</div>
        <div class="text-base font-bold mt-1 ${c.color || "text-white"}">${c.value}</div>
      </div>`)
    .join("");
}

async function runInference() {
  if (!selectedFile && !selectedSample) {
    alert("请先上传本地音频文件或从下拉框选择内置示例音频！");
    return;
  }

  const btn = document.getElementById("runBtn");
  btn.disabled = true;
  btn.textContent = "GPU 实时推理中...";

  try {
    const api = getApiBase();
    const fd = new FormData();
    if (selectedFile) fd.append("file", selectedFile);
    else fd.append("sample_filename", selectedSample);

    let endpoint = "/api/detect";
    if (currentTab === "detect") {
      fd.append("model", document.getElementById("cmModelSelect").value);
    } else if (currentTab === "sasv") {
      const spk = document.getElementById("targetSpeakerSelect").value;
      if (!spk) { alert("请先注册或选择目标说话人（可点击绿色按钮一键注册 LA_0015）！"); return; }
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

    const resp = await fetch(`${api}${endpoint}`, { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) { alert("推理出错: " + (data.detail || resp.statusText)); return; }

    document.getElementById("emptyState").classList.add("hidden");
    document.getElementById("resultContent").classList.remove("hidden");
    document.getElementById("rawJsonPre").textContent = JSON.stringify(data, null, 2);

    const banner = document.getElementById("verdictBanner");
    const vTitle = document.getElementById("verdictTitle");
    const vSub = document.getElementById("verdictSub");
    const vLat = document.getElementById("verdictLatency");

    if (currentTab === "detect") {
      const isFake = data.verdict === "spoof";
      banner.className = isFake
        ? "rounded-xl p-4 border bg-rose-950/60 border-rose-700 text-rose-100 flex flex-wrap items-center justify-between gap-3"
        : "rounded-xl p-4 border bg-emerald-950/60 border-emerald-700 text-emerald-100 flex flex-wrap items-center justify-between gap-3";
      vSub.textContent = `非特定人伪造检测 · 模型: ${data.model} (${data.backend})`;
      vTitle.textContent = data.verdict_zh;
      vLat.textContent = `${data.latency_ms} ms`;
      renderMetricCards([
        { label: "伪造概率", value: `${data.spoof_probability}%`, color: isFake ? "text-rose-400" : "text-slate-200" },
        { label: "真声概率", value: `${data.bonafide_probability}%`, color: !isFake ? "text-emerald-400" : "text-slate-200" },
        { label: "Bonafide Logit", value: data.bonafide_logit },
        { label: "音频时长", value: `${data.duration_sec} s` },
      ]);
      renderWaveform(data.waveform_preview);
      renderTraceChart(null);
    } else if (currentTab === "sasv") {
      const ok = data.accepted;
      banner.className = ok
        ? "rounded-xl p-4 border bg-emerald-950/60 border-emerald-700 text-emerald-100 flex flex-wrap items-center justify-between gap-3"
        : "rounded-xl p-4 border bg-rose-950/60 border-rose-700 text-rose-100 flex flex-wrap items-center justify-between gap-3";
      vSub.textContent = `特定人防伪声纹验证 (ECAPA-TDNN + ${data.cm_model})`;
      vTitle.textContent = data.verdict_zh;
      vLat.textContent = `${data.latency_ms} ms`;
      renderMetricCards([
        { label: "声纹相似度 (Cos)", value: data.speaker_similarity, color: data.speaker_similarity >= 0.45 ? "text-emerald-400" : "text-amber-400" },
        { label: "防伪真声概率", value: `${data.cm_bonafide_probability}%`, color: data.cm_bonafide_probability >= 50 ? "text-emerald-400" : "text-rose-400" },
        { label: "SASV 融合得分", value: data.sasv_fusion_score, color: "text-cyan-300" },
        { label: "核验状态", value: ok ? "ACCEPT" : "REJECT", color: ok ? "text-emerald-400" : "text-rose-400" },
      ]);
      renderWaveform(data.waveform_preview);
      renderTraceChart(null);
    } else if (currentTab === "trace") {
      banner.className = "rounded-xl p-4 border bg-sky-950/60 border-sky-700 text-sky-100 flex flex-wrap items-center justify-between gap-3";
      vSub.textContent = `伪造音频算法取证溯源 · ${data.model}`;
      vTitle.textContent = `${data.predicted_algorithm} · ${data.predicted_description}`;
      vLat.textContent = `${data.latency_ms} ms`;
      renderMetricCards([
        { label: "Top-1 算法", value: data.predicted_algorithm, color: "text-cyan-300" },
        { label: "技术大类", value: data.predicted_category },
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
        ? "rounded-xl p-4 border bg-rose-950/60 border-rose-700 text-rose-100 flex flex-wrap items-center justify-between gap-3"
        : "rounded-xl p-4 border bg-emerald-950/60 border-emerald-700 text-emerald-100 flex flex-wrap items-center justify-between gap-3";
      vSub.textContent = "全流程综合取证结论";
      vTitle.textContent = sv ? sv.verdict_zh : det.verdict_zh;
      vLat.textContent = `${data.total_latency_ms} ms`;
      renderMetricCards([
        { label: "伪造概率", value: `${det.spoof_probability}%`, color: isFake ? "text-rose-400" : "text-emerald-400" },
        { label: "声纹相似度", value: sv ? sv.speaker_similarity : "未启用", color: "text-cyan-300" },
        { label: "溯源 Top-1", value: tr ? tr.predicted_algorithm : "真声跳过", color: "text-amber-300" },
        { label: "溯源置信度", value: tr ? `${tr.top1_confidence_pct}%` : "-" },
      ]);
      renderWaveform(det.waveform_preview);
      renderTraceChart(tr ? tr.top_predictions : null);
    }
  } finally {
    btn.disabled = false;
    btn.textContent = "开始 GPU 实时检测分析";
  }
}

// 绑定事件
document.addEventListener("DOMContentLoaded", () => {
  const saved = localStorage.getItem("asv_api_base");
  if (saved) document.getElementById("apiBaseInput").value = saved;
  document.getElementById("swaggerLink").href = `${getApiBase()}/docs`;

  const fileInput = document.getElementById("audioFileInput");
  const sampleSel = document.getElementById("sampleSelect");
  const player = document.getElementById("audioPlayer");
  const playerWrap = document.getElementById("audioPlayerWrap");
  const lbl = document.getElementById("selectedAudioLabel");

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      selectedFile = e.target.files[0];
      selectedSample = "";
      sampleSel.value = "";
      lbl.textContent = selectedFile.name;
      player.src = URL.createObjectURL(selectedFile);
      playerWrap.classList.remove("hidden");
      updateSnippets();
    }
  });

  sampleSel.addEventListener("change", (e) => {
    selectedSample = e.target.value;
    if (selectedSample) {
      selectedFile = null;
      fileInput.value = "";
      lbl.textContent = selectedSample;
      player.src = `${getApiBase()}/api/samples/${encodeURIComponent(selectedSample)}`;
      playerWrap.classList.remove("hidden");
    } else {
      lbl.textContent = "未选择音频";
      playerWrap.classList.add("hidden");
    }
    updateSnippets();
  });

  ["cmModelSelect", "traceHeadSelect", "targetSpeakerSelect"].forEach((id) => {
    document.getElementById(id).addEventListener("change", updateSnippets);
  });

  initData();
});
