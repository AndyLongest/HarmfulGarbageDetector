const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const dropPrompt = document.querySelector("#dropPrompt");
const previewImage = document.querySelector("#previewImage");
const detectButton = document.querySelector("#detectButton");
const resetButton = document.querySelector("#resetButton");
const confidence = document.querySelector("#confidence");
const confidenceValue = document.querySelector("#confidenceValue");
const emptyState = document.querySelector("#emptyState");
const loadingState = document.querySelector("#loadingState");
const resultState = document.querySelector("#resultState");
let selectedFile = null;

confidence.addEventListener("input", () => {
  confidenceValue.value = Number(confidence.value).toFixed(2);
});

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
["dragenter", "dragover"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", event => setFile(event.dataTransfer.files[0]));
resetButton.addEventListener("click", event => {
  event.preventDefault();
  selectedFile = null;
  fileInput.value = "";
  previewImage.src = "";
  previewImage.hidden = true;
  dropPrompt.hidden = false;
  resetButton.hidden = true;
  detectButton.disabled = true;
  showState("empty");
});

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  previewImage.src = URL.createObjectURL(file);
  previewImage.hidden = false;
  dropPrompt.hidden = true;
  resetButton.hidden = false;
  detectButton.disabled = false;
}

function showState(state) {
  emptyState.hidden = state !== "empty";
  loadingState.hidden = state !== "loading";
  resultState.hidden = state !== "result";
}

detectButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  showState("loading");
  detectButton.disabled = true;
  const form = new FormData();
  form.append("image", selectedFile);
  form.append("confidence", confidence.value);

  try {
    const response = await fetch("/api/detect", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "检测失败");
    renderResult(data);
    showState("result");
  } catch (error) {
    alert(`检测失败：${error.message}`);
    showState("empty");
  } finally {
    detectButton.disabled = false;
  }
});

function renderResult(data) {
  const alertBox = document.querySelector("#alertBox");
  const alertTitle = document.querySelector("#alertTitle");
  const alertText = document.querySelector("#alertText");
  const alertSymbol = alertBox.querySelector(".alert-symbol");
  document.querySelector("#resultImage").src = data.annotated_image;
  const resized = data.original_size.join("×") !== data.processed_size.join("×");
  const sizeText = resized ? ` · ${data.original_size.join("×")} → ${data.processed_size.join("×")}` : "";
  document.querySelector("#latency").textContent =
    `预处理 ${data.preprocessing_ms} ms · 推理 ${data.inference_ms} ms${sizeText}`;
  document.querySelector("#totalCount").textContent = `共发现 ${data.total} 个目标`;

  alertBox.className = `alert-box ${data.hazardous ? "danger" : "safe"}`;
  alertSymbol.textContent = data.hazardous ? "!" : "✓";
  alertTitle.textContent = data.hazardous ? "发现有害垃圾，请谨慎处理" : "未发现有害垃圾";
  if (data.hazardous) {
    const summary = Object.entries(data.hazardous_counts).map(([name, count]) => `${name} × ${count}`).join("、");
    alertText.textContent = `重点目标：${summary}。请勿直接接触或混入普通垃圾。`;
  } else if (data.total) {
    alertText.textContent = "已识别到普通垃圾，建议按照当地分类要求投放。";
  } else {
    alertText.textContent = "当前阈值下未识别到垃圾目标，可尝试降低检测灵敏度阈值。";
  }

  const list = document.querySelector("#detectionList");
  const sorted = [...data.detections].sort((a, b) => Number(b.hazardous) - Number(a.hazardous) || b.confidence - a.confidence);
  list.innerHTML = sorted.length ? sorted.map(item => `
    <div class="detection-item ${item.hazardous ? "danger" : ""}">
      <div><strong>${escapeHtml(item.display_name)}</strong><small>${escapeHtml(item.name)}${item.hazardous ? " · 有害垃圾" : ""}</small></div>
      <b>${Math.round(item.confidence * 100)}%</b>
    </div>`).join("") : '<div class="detection-item"><div><strong>未发现目标</strong><small>请尝试其他图片或降低阈值</small></div></div>';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  })[char]);
}
