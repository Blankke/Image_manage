"use strict";

const byId = (id) => document.getElementById(id);
const state = {
  files: [],
  sourceImage: null,
  geometryImage: null,
  sourceUrl: null,
  resultImage: null,
  resultUrl: null,
  resultBlob: null,
  results: {},
  resultUrls: {},
  resultBlobs: {},
  resultDiagnostics: {},
  activeResult: "fidelity",
  actualPixels: false,
  mode: "corners",
  corners: [[0.04, 0.04], [0.96, 0.04], [0.96, 0.96], [0.04, 0.96]],
  mesh: [],
  dragging: null,
};

const canvas = byId("editorCanvas");
const context = canvas.getContext("2d", { alpha: false });

function numberValue(id) {
  const value = Number(byId(id).value);
  if (!Number.isFinite(value)) throw new Error(`${id} 不是有效数值`);
  return value;
}

function setStatus(message, mode = "ready", progress = 0) {
  byId("statusText").textContent = message;
  byId("statusDot").className = `status-dot ${mode}`;
  byId("progress").value = progress;
}

function setBusy(busy, message) {
  byId("restoreButton").disabled = busy || !state.files.length;
  byId("autoDetect").disabled = busy || !state.files.length;
  byId("calibrateButton").disabled = busy;
  if (busy) setStatus(message, "busy", 35);
}

function updateFiles(files) {
  Object.values(state.resultUrls).forEach((url) => URL.revokeObjectURL(url));
  state.results = {}; state.resultUrls = {}; state.resultBlobs = {}; state.resultDiagnostics = {};
  state.resultImage = null; state.resultUrl = null; state.resultBlob = null;
  byId("resultToolbar").classList.add("hidden");
  byId("downloadButton").disabled = true;
  state.files = Array.from(files).slice(0, 20);
  const summary = byId("fileSummary");
  summary.replaceChildren();
  if (!state.files.length) {
    const item = document.createElement("li");
    item.textContent = "尚未选择照片";
    summary.append(item);
    byId("fileCount").textContent = "0 张";
    return;
  }
  state.files.forEach((file, index) => {
    const item = document.createElement("li");
    item.textContent = `${String(index + 1).padStart(2, "0")}  ${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MiB`;
    summary.append(item);
  });
  byId("fileCount").textContent = `${state.files.length} 张`;
  byId("restoreButton").disabled = false;
  byId("autoDetect").disabled = false;
  byId("resetGeometry").disabled = false;
  loadSourceImage(state.files[0]);
  setStatus(state.files.length > 1 ? `已选择 ${state.files.length} 张，将启用多帧融合` : "单张经典恢复模式", "ready");
}

function loadSourceImage(file) {
  if (state.sourceUrl) URL.revokeObjectURL(state.sourceUrl);
  state.sourceUrl = URL.createObjectURL(file);
  state.geometryImage = null;
  const image = new Image();
  image.onload = () => {
    state.sourceImage = image;
    byId("canvasStage").classList.remove("empty");
    byId("emptyHint").classList.add("hidden");
    resetCorners();
    resetMesh();
    drawCanvas();
  };
  image.src = state.sourceUrl;
}

function displayedImage() {
  if (state.mode === "result") return resultByName(state.activeResult);
  if (state.mode === "mesh" && state.resultImage) return state.resultImage;
  if (state.mode === "corners" && state.geometryImage) return state.geometryImage;
  return state.sourceImage;
}

function resultByName(name) {
  if (name === "original") return state.sourceImage;
  return state.results[name] || null;
}

function drawCanvas() {
  const image = displayedImage();
  if (!image) return;
  if (state.mode === "result" && byId("comparisonMode").value !== "single" && state.results.geometry) {
    drawComparisonCanvas();
    return;
  }
  const maxWidth = 1200;
  const maxHeight = 820;
  const scale = state.actualPixels ? 1 : Math.min(1, maxWidth / image.naturalWidth, maxHeight / image.naturalHeight);
  canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  if (state.mode === "corners") drawCorners();
  if (state.mode === "mesh") drawMesh();
}

function drawComparisonCanvas() {
  let baseline = state.results.geometry;
  let compared = state.results.fidelity;
  let baselineLabel = "Geometry";
  let comparedLabel = "Fidelity";
  if (state.activeResult === "original" || state.activeResult === "geometry") {
    baseline = state.sourceImage;
    compared = state.results.geometry;
    baselineLabel = "Original";
    comparedLabel = "Geometry";
  } else if (state.activeResult === "ai") {
    baseline = state.results.fidelity || state.results.geometry;
    compared = state.results.ai;
    baselineLabel = state.results.fidelity ? "Fidelity" : "Geometry";
    comparedLabel = "AI Enhanced";
  }
  if (!baseline || !compared) return;
  const comparisonMode = byId("comparisonMode").value;
  const maxWidth = 1200;
  const maxHeight = 820;
  const sideFactor = comparisonMode === "side" ? 2 : 1;
  const scale = state.actualPixels ? 1 : Math.min(1, maxWidth / (compared.naturalWidth * sideFactor), maxHeight / compared.naturalHeight);
  const width = Math.max(1, Math.round(compared.naturalWidth * scale));
  const height = Math.max(1, Math.round(compared.naturalHeight * scale));
  canvas.width = width * sideFactor;
  canvas.height = height;
  if (comparisonMode === "side") {
    context.drawImage(baseline, 0, 0, width, height);
    context.drawImage(compared, width, 0, width, height);
    drawLabel(baselineLabel, 12, 24);
    drawLabel(comparedLabel, width + 12, 24);
    return;
  }
  context.drawImage(baseline, 0, 0, width, height);
  const split = numberValue("splitPosition");
  context.save();
  context.beginPath(); context.rect(0, 0, width * split, height); context.clip();
  context.drawImage(compared, 0, 0, width, height);
  context.restore();
  context.strokeStyle = "#e8ff47"; context.lineWidth = 2;
  context.beginPath(); context.moveTo(width * split, 0); context.lineTo(width * split, height); context.stroke();
  drawLabel(comparedLabel, 12, 24);
  drawLabel(baselineLabel, Math.max(12, width - 104), 24);
}

function drawLabel(label, x, y) {
  context.font = "700 11px ui-monospace";
  context.fillStyle = "rgba(0,0,0,.72)"; context.fillRect(x - 6, y - 16, context.measureText(label).width + 12, 22);
  context.fillStyle = "#e8ff47"; context.fillText(label, x, y);
}

function drawCorners() {
  const points = state.corners.map(([x, y]) => [x * canvas.width, y * canvas.height]);
  context.save();
  context.strokeStyle = "#e8ff47";
  context.lineWidth = Math.max(2, canvas.width / 500);
  context.shadowColor = "rgba(0,0,0,.8)";
  context.shadowBlur = 6;
  context.beginPath();
  points.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
  context.closePath();
  context.stroke();
  points.forEach(([x, y], index) => {
    context.fillStyle = "#0b0e11";
    context.strokeStyle = "#e8ff47";
    context.lineWidth = 3;
    context.beginPath();
    context.arc(x, y, 9, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.fillStyle = "#e8ff47";
    context.font = "700 10px ui-monospace";
    context.fillText(String(index + 1), x + 13, y - 11);
  });
  context.restore();
}

function drawMesh() {
  if (!state.mesh.length) resetMesh();
  context.save();
  context.strokeStyle = "rgba(85,220,231,.82)";
  context.lineWidth = 1.35;
  state.mesh.forEach((row) => {
    context.beginPath();
    row.forEach(([x, y], index) => index ? context.lineTo(x * canvas.width, y * canvas.height) : context.moveTo(x * canvas.width, y * canvas.height));
    context.stroke();
  });
  for (let column = 0; column < state.mesh[0].length; column += 1) {
    context.beginPath();
    state.mesh.forEach((row, index) => index ? context.lineTo(row[column][0] * canvas.width, row[column][1] * canvas.height) : context.moveTo(row[column][0] * canvas.width, row[column][1] * canvas.height));
    context.stroke();
  }
  state.mesh.flat().forEach(([x, y]) => {
    context.fillStyle = "#55dce7";
    context.beginPath();
    context.arc(x * canvas.width, y * canvas.height, 4.2, 0, Math.PI * 2);
    context.fill();
  });
  context.restore();
}

function resetCorners() {
  state.corners = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]];
  drawCanvas();
}

function regularGrid(rows, columns) {
  return Array.from({ length: rows }, (_, row) => Array.from({ length: columns }, (_, column) => [column / (columns - 1), row / (rows - 1)]));
}

function resetMesh() {
  const rows = Math.max(2, Math.min(15, numberValue("meshRows")));
  const columns = Math.max(2, Math.min(15, numberValue("meshColumns")));
  state.mesh = regularGrid(rows, columns);
  drawCanvas();
}

function generateCurvedMesh() {
  const rows = numberValue("meshRows");
  const columns = numberValue("meshColumns");
  const horizontal = numberValue("horizontalCurve");
  const vertical = numberValue("verticalCurve");
  state.mesh = regularGrid(rows, columns).map((row) => row.map(([x, y]) => {
    const unitX = x * 2 - 1;
    const unitY = y * 2 - 1;
    return [
      Math.max(0, Math.min(1, x + vertical * (1 - unitY ** 2) * unitX * (1 - unitX ** 2))),
      Math.max(0, Math.min(1, y + horizontal * (1 - unitX ** 2) * unitY * (1 - unitY ** 2))),
    ];
  }));
  byId("meshEnabled").checked = true;
  changeMode("mesh");
}

function changeMode(mode) {
  state.mode = mode;
  ["editCorners", "editMesh", "viewResult"].forEach((id) => byId(id).classList.remove("active"));
  const activeId = mode === "corners" ? "editCorners" : mode === "mesh" ? "editMesh" : "viewResult";
  byId(activeId).classList.add("active");
  if (mode === "mesh" && !state.resultImage) {
    setStatus("Mesh 作用于透视输出；先恢复一次可获得准确背景", "ready", 0);
  }
  drawCanvas();
}

function pointerLocation(event) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height];
}

function nearestHandle(point) {
  const threshold = 20 / Math.min(canvas.getBoundingClientRect().width, canvas.getBoundingClientRect().height);
  if (state.mode === "corners") {
    let best = null;
    state.corners.forEach((candidate, index) => {
      const distance = Math.hypot(candidate[0] - point[0], candidate[1] - point[1]);
      if (distance < threshold && (!best || distance < best.distance)) best = { type: "corner", index, distance };
    });
    return best;
  }
  if (state.mode === "mesh") {
    let best = null;
    state.mesh.forEach((row, rowIndex) => row.forEach((candidate, columnIndex) => {
      const distance = Math.hypot(candidate[0] - point[0], candidate[1] - point[1]);
      if (distance < threshold && (!best || distance < best.distance)) best = { type: "mesh", rowIndex, columnIndex, distance };
    }));
    return best;
  }
  return null;
}

function lensSettings() {
  return {
    enabled: byId("lensEnabled").checked,
    model: byId("lensModel").value,
    focal_x: numberValue("focalX"), focal_y: numberValue("focalY"),
    principal_x: numberValue("principalX"), principal_y: numberValue("principalY"),
    k1: numberValue("k1"), k2: numberValue("k2"), p1: numberValue("p1"), p2: numberValue("p2"),
    k3: numberValue("k3"), k4: numberValue("k4"),
    optimize_camera_matrix: byId("optimizeCamera").value === "true",
    crop_balance: 0, crop_to_valid: false,
  };
}

function aiSettings(enabled) {
  return {
    enabled,
    manifest_id: byId("aiModel").value,
    strength: numberValue("aiStrength"),
    denoise_strength: numberValue("aiDenoise"),
    output_scale: numberValue("aiOutscale"),
    blend_strength: 1,
  };
}

function artifactOverrides() {
  const overrides = {};
  const demoirePolicy = byId("demoirePolicy").value;
  if (demoirePolicy === "off") {
    overrides.demoire = { enabled: false };
  } else if (demoirePolicy === "strong") {
    overrides.demoire = {
      enabled: true,
      params: {
        mode: "joint_edge_aware", strength: 1, chroma_radius: 3.2,
        luma_sigma_color: 0.08, edge_protection: 0.65,
        chroma_relative_strength: 0.8, minimum_filter_weight: 0.45,
      },
    };
  }

  const dehaloPolicy = byId("dehaloPolicy").value;
  if (dehaloPolicy === "off") {
    overrides.dehalo = { enabled: false };
  } else if (dehaloPolicy === "gated") {
    overrides.dehalo = { enabled: true, params: { auto_gate: true } };
  } else if (dehaloPolicy === "strong") {
    overrides.dehalo = {
      enabled: true,
      params: {
        auto_gate: true, strength: 0.5, max_correction: 0.1,
        max_scene_median: 0.45, max_highlight_area: 0.35,
      },
    };
  }
  return overrides;
}

function restoreSettings(outputVariant = "fidelity") {
  const aiEnabled = outputVariant === "ai_enhanced";
  const settings = {
    preset: byId("preset").value,
    processing_mode: aiEnabled ? "ai_enhanced" : "fidelity",
    output_variant: outputVariant,
    ai: aiSettings(aiEnabled),
    operator_overrides: artifactOverrides(),
    corners: state.corners,
    ratio_mode: byId("ratioMode").value,
    custom_ratio: numberValue("customRatio"),
    lens: lensSettings(),
    mesh: {
      enabled: byId("meshEnabled").checked,
      rows: state.mesh.length,
      columns: state.mesh[0]?.length || 0,
      control_points: state.mesh,
      strength: 1,
      interpolation: "cubic",
      border_mode: "replicate",
    },
    fusion: {
      alignment: "auto", reference_index: -1, max_frames: 8,
      max_alignment_dimension: 1400, minimum_overlap: 0.55,
      minimum_alignment_score: 0.12, outlier_threshold: 0.1,
      exposure_compensation: true,
    },
  };
  return settings;
}

async function apiError(response) {
  try { const payload = await response.json(); return payload.error || `HTTP ${response.status}`; }
  catch { return `HTTP ${response.status}`; }
}

async function autoDetect() {
  if (!state.files.length) return;
  setBusy(true, "正在分析屏幕边界");
  const form = new FormData();
  form.append("files", state.files[0], state.files[0].name);
  form.append("settings", JSON.stringify({ lens: lensSettings() }));
  try {
    const response = await fetch("/api/v1/detect", { method: "POST", body: form });
    if (!response.ok) throw new Error(await apiError(response));
    const payload = await response.json();
    if (!payload.candidates.length) throw new Error("未检测到可靠四边形，请手动拖动四角");
    if (payload.corrected_preview?.data_url) {
      const corrected = new Image();
      await new Promise((resolve, reject) => {
        corrected.onload = resolve; corrected.onerror = reject; corrected.src = payload.corrected_preview.data_url;
      });
      state.geometryImage = corrected;
    } else {
      state.geometryImage = null;
    }
    state.corners = payload.candidates[0].corners;
    changeMode("corners");
    drawCanvas();
    setStatus(`四角检测完成，置信度 ${(payload.candidates[0].confidence * 100).toFixed(1)}%`, "ready", 100);
  } catch (error) { setStatus(error.message, "", 0); }
  finally { setBusy(false, ""); }
}

function decodeDiagnostics(value) {
  if (!value) return null;
  const padded = value.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - value.length % 4) % 4);
  const bytes = Uint8Array.from(atob(padded), (character) => character.charCodeAt(0));
  return JSON.parse(new TextDecoder().decode(bytes));
}

async function restore() {
  if (!state.files.length) return;
  const wantsAi = byId("processingMode").value === "ai_enhanced";
  if (wantsAi && !byId("aiModel").value) {
    setStatus("AI Enhanced 模式尚未选择可用本地模型", "", 0);
    return;
  }
  setBusy(true, state.files.length > 1 ? "正在对齐并融合真实观测" : "正在生成分阶段对比");
  byId("downloadButton").disabled = true;
  Object.values(state.resultUrls).forEach((url) => URL.revokeObjectURL(url));
  state.results = {}; state.resultUrls = {}; state.resultBlobs = {}; state.resultDiagnostics = {};
  try {
    await requestVariant("geometry");
    setStatus("几何结果完成，正在忠实恢复", "busy", 48);
    await requestVariant("fidelity");
    if (wantsAi) {
      setStatus("忠实恢复完成，正在运行本地 AI 增强", "busy", 78);
      await requestVariant("ai_enhanced");
    }
    state.activeResult = wantsAi ? "ai" : "fidelity";
    state.resultImage = resultByName(state.activeResult);
    state.resultBlob = state.resultBlobs[state.activeResult];
    state.resultUrl = state.resultUrls[state.activeResult];
    byId("diagnostics").textContent = JSON.stringify(state.resultDiagnostics[state.activeResult], null, 2);
    byId("downloadButton").disabled = false;
    byId("resultToolbar").classList.remove("hidden");
    const aiTab = document.querySelector('[data-result="ai"]');
    aiTab.disabled = !state.results.ai;
    selectResult(state.activeResult);
    changeMode("result");
    const image = state.resultImage;
    setStatus(`分阶段结果完成 · ${image.naturalWidth}×${image.naturalHeight}`, "ready", 100);
  } catch (error) {
    setStatus(error.message || "恢复失败", "", 0);
    byId("diagnostics").textContent = String(error.stack || error);
  } finally { setBusy(false, ""); }
}

async function requestVariant(variant) {
  const form = new FormData();
  state.files.forEach((file) => form.append("files", file, file.name));
  form.append("settings", JSON.stringify(restoreSettings(variant)));
  const response = await fetch("/api/v1/restore", { method: "POST", body: form });
  if (!response.ok) throw new Error(await apiError(response));
  const blob = await response.blob();
  const key = variant === "ai_enhanced" ? "ai" : variant;
  const url = URL.createObjectURL(blob);
  const image = new Image();
  await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = url; });
  state.results[key] = image;
  state.resultUrls[key] = url;
  state.resultBlobs[key] = blob;
  state.resultDiagnostics[key] = decodeDiagnostics(response.headers.get("X-ScreenRestore-Diagnostics"));
}

function selectResult(name) {
  if (!resultByName(name)) return;
  state.activeResult = name;
  state.resultImage = resultByName(name);
  if (name !== "original") {
    state.resultBlob = state.resultBlobs[name];
    state.resultUrl = state.resultUrls[name];
    byId("diagnostics").textContent = JSON.stringify(state.resultDiagnostics[name], null, 2);
    byId("downloadButton").disabled = false;
  } else {
    state.resultBlob = null;
    state.resultUrl = null;
    byId("downloadButton").disabled = true;
    byId("diagnostics").textContent = "Original 为只读输入；请选择 Geometry、Fidelity 或 AI Enhanced 查看对应诊断。";
  }
  document.querySelectorAll("[data-result]").forEach((button) => button.classList.toggle("active", button.dataset.result === name));
  drawCanvas();
}

async function calibrate() {
  const files = Array.from(byId("calibrationFiles").files || []);
  if (files.length < 3) { setStatus("镜头标定至少选择 3 张棋盘格图片", "", 0); return; }
  setBusy(true, "正在检测棋盘格并标定镜头");
  const form = new FormData();
  files.forEach((file) => form.append("calibration_files", file, file.name));
  form.append("settings", JSON.stringify({
    board_columns: numberValue("boardColumns"), board_rows: numberValue("boardRows"),
    square_size: numberValue("squareSize"), min_views: Math.max(3, Math.min(5, files.length)),
    model: byId("lensModel").value,
  }));
  try {
    const response = await fetch("/api/v1/calibrate", { method: "POST", body: form });
    if (!response.ok) throw new Error(await apiError(response));
    const payload = await response.json();
    const lens = payload.lens;
    const mapping = { focalX: "focal_x", focalY: "focal_y", principalX: "principal_x", principalY: "principal_y", k1: "k1", k2: "k2", p1: "p1", p2: "p2", k3: "k3", k4: "k4" };
    Object.entries(mapping).forEach(([id, key]) => { byId(id).value = Number(lens[key]).toPrecision(8); });
    byId("lensEnabled").checked = true;
    byId("diagnostics").textContent = JSON.stringify(payload, null, 2);
    setStatus(`标定完成 · ${payload.used_views} 张有效 · RMS ${payload.rms_error.toFixed(3)} px`, "ready", 100);
  } catch (error) { setStatus(error.message, "", 0); }
  finally { setBusy(false, ""); }
}

function downloadResult() {
  if (!state.resultBlob) return;
  const link = document.createElement("a");
  const baseName = state.files[0]?.name.replace(/\.[^.]+$/, "") || "screenrestore";
  link.href = state.resultUrl;
  const labels = { geometry: "几何", fidelity: "忠实恢复", ai: "AI增强" };
  link.download = `${baseName}_${labels[state.activeResult] || "恢复"}.png`;
  link.click();
}

async function loadModels() {
  const select = byId("aiModel");
  select.replaceChildren();
  try {
    const response = await fetch("/api/v1/models");
    if (!response.ok) throw new Error(await apiError(response));
    const payload = await response.json();
    const models = (payload.models || []).filter((model) => model.role === "enhancement");
    models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.available ? model.id : "";
      option.disabled = !model.available;
      option.textContent = `${model.name} · ${model.task}${model.available ? "" : `（不可用：${model.status}）`}`;
      select.append(option);
    });
    const firstAvailable = models.find((model) => model.available);
    if (firstAvailable) {
      select.value = firstAvailable.id;
      setStatus(`已发现本地 AI 模型：${firstAvailable.name}`, "ready", 0);
    } else {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = models.length ? "增强模型均不可用" : "未发现增强模型";
      select.prepend(option);
      select.value = "";
    }
  } catch (error) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "模型目录读取失败";
    select.append(option);
    setStatus(error.message, "", 0);
  }
}

function updateProcessingMode() {
  const aiEnabled = byId("processingMode").value === "ai_enhanced";
  byId("aiControls").classList.toggle("hidden", !aiEnabled);
}

byId("imageFiles").addEventListener("change", (event) => updateFiles(event.target.files));
const dropZone = byId("dropZone");
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); }));
dropZone.addEventListener("drop", (event) => updateFiles(event.dataTransfer.files));
byId("ratioMode").addEventListener("change", () => byId("customRatioRow").classList.toggle("hidden", byId("ratioMode").value !== "custom"));
byId("lensEnabled").addEventListener("change", () => {
  state.geometryImage = null;
  drawCanvas();
  if (byId("lensEnabled").checked) setStatus("镜头参数已启用；点击“自动四角”生成校正预览", "ready", 0);
});
byId("editCorners").addEventListener("click", () => changeMode("corners"));
byId("editMesh").addEventListener("click", () => changeMode("mesh"));
byId("viewResult").addEventListener("click", () => changeMode("result"));
byId("autoDetect").addEventListener("click", autoDetect);
byId("restoreButton").addEventListener("click", restore);
byId("downloadButton").addEventListener("click", downloadResult);
byId("calibrateButton").addEventListener("click", calibrate);
byId("applyCurve").addEventListener("click", generateCurvedMesh);
byId("meshRows").addEventListener("change", resetMesh);
byId("meshColumns").addEventListener("change", resetMesh);
byId("resetGeometry").addEventListener("click", () => state.mode === "mesh" ? resetMesh() : resetCorners());
byId("processingMode").addEventListener("change", updateProcessingMode);
[["aiStrength", "aiStrengthValue"], ["aiDenoise", "aiDenoiseValue"]].forEach(([inputId, outputId]) => {
  byId(inputId).addEventListener("input", () => { byId(outputId).value = byId(inputId).value; });
});
document.querySelectorAll("[data-result]").forEach((button) => {
  button.addEventListener("click", () => selectResult(button.dataset.result));
});
byId("comparisonMode").addEventListener("change", () => {
  byId("splitControl").classList.toggle("hidden", byId("comparisonMode").value !== "split");
  drawCanvas();
});
byId("splitPosition").addEventListener("input", drawCanvas);
byId("actualPixels").addEventListener("click", () => {
  state.actualPixels = !state.actualPixels;
  byId("actualPixels").classList.toggle("active", state.actualPixels);
  byId("actualPixels").textContent = state.actualPixels ? "适应" : "100%";
  drawCanvas();
});

canvas.addEventListener("pointerdown", (event) => {
  state.dragging = nearestHandle(pointerLocation(event));
  if (state.dragging) canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.dragging) return;
  const [rawX, rawY] = pointerLocation(event);
  const x = Math.max(0, Math.min(1, rawX));
  const y = Math.max(0, Math.min(1, rawY));
  if (state.dragging.type === "corner") state.corners[state.dragging.index] = [x, y];
  else state.mesh[state.dragging.rowIndex][state.dragging.columnIndex] = [x, y];
  drawCanvas();
});
["pointerup", "pointercancel"].forEach((name) => canvas.addEventListener(name, () => { state.dragging = null; }));
window.addEventListener("resize", drawCanvas);
updateProcessingMode();
loadModels();
