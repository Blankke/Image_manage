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
  if (state.mode === "result" && state.resultImage) return state.resultImage;
  if (state.mode === "mesh" && state.resultImage) return state.resultImage;
  if (state.mode === "corners" && state.geometryImage) return state.geometryImage;
  return state.sourceImage;
}

function drawCanvas() {
  const image = displayedImage();
  if (!image) return;
  const maxWidth = 1200;
  const maxHeight = 820;
  const scale = Math.min(1, maxWidth / image.naturalWidth, maxHeight / image.naturalHeight);
  canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  if (state.mode === "corners") drawCorners();
  if (state.mode === "mesh") drawMesh();
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

function restoreSettings() {
  const settings = {
    preset: byId("preset").value,
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
  setBusy(true, state.files.length > 1 ? "正在对齐并融合真实观测" : "正在运行恢复流水线");
  byId("downloadButton").disabled = true;
  const form = new FormData();
  state.files.forEach((file) => form.append("files", file, file.name));
  form.append("settings", JSON.stringify(restoreSettings()));
  try {
    const response = await fetch("/api/v1/restore", { method: "POST", body: form });
    if (!response.ok) throw new Error(await apiError(response));
    const blob = await response.blob();
    if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
    state.resultBlob = blob;
    state.resultUrl = URL.createObjectURL(blob);
    const image = new Image();
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = state.resultUrl; });
    state.resultImage = image;
    const diagnostics = decodeDiagnostics(response.headers.get("X-ScreenRestore-Diagnostics"));
    byId("diagnostics").textContent = JSON.stringify(diagnostics, null, 2);
    byId("downloadButton").disabled = false;
    changeMode("result");
    setStatus(`恢复完成 · ${image.naturalWidth}×${image.naturalHeight}`, "ready", 100);
  } catch (error) {
    setStatus(error.message || "恢复失败", "", 0);
    byId("diagnostics").textContent = String(error.stack || error);
  } finally { setBusy(false, ""); }
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
  link.download = `${baseName}_恢复.png`;
  link.click();
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
