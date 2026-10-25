const form = document.querySelector("#ask-form");
const documentSelect = document.querySelector("#document");
const questionInput = document.querySelector("#question");
const askButton = document.querySelector("#ask");
const status = document.querySelector("#status");
const errorBox = document.querySelector("#error");
const result = document.querySelector("#result");
const answer = document.querySelector("#answer");
const citations = document.querySelector("#citations");
const noCitations = document.querySelector("#no-citations");
const fileInput = document.querySelector("#file-input");
const folderInput = document.querySelector("#folder-input");
const addFiles = document.querySelector("#add-files");
const addFolder = document.querySelector("#add-folder");
const uploadButton = document.querySelector("#upload");
const selection = document.querySelector("#selection");
const fileList = document.querySelector("#file-list");
const knowledge = document.querySelector("#knowledge");

const SUPPORTED = [".pdf", ".docx"];
let pending = [];
let asking = false;
let askAllowed = false;
let pollTimer = 0;

function showError(message) {
  errorBox.hidden = false;
  errorBox.textContent = message;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function errorMessage(payload, statusCode) {
  if (payload && typeof payload.detail === "string") return payload.detail;
  if (payload && Array.isArray(payload.detail)) {
    return payload.detail.map((item) => item.msg || "Invalid request.").join(" ");
  }
  return "The request failed (" + statusCode + ").";
}

function refreshAskButton() {
  askButton.disabled = asking || !askAllowed;
}

function addOption(value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  documentSelect.append(option);
}

function renderCitations(items) {
  citations.replaceChildren();
  const list = Array.isArray(items) ? items : [];
  noCitations.hidden = list.length > 0;
  for (const item of list) {
    const li = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = item.document || "Document";
    li.append(title);
    const meta = [];
    if (item.page != null) {
      let page = "p. " + item.page;
      if (item.page_label && String(item.page_label) !== String(item.page)) {
        page += " (printed " + item.page_label + ")";
      }
      meta.push(page);
    }
    if (item.section) meta.push(item.section);
    if (item.label) meta.push(item.label);
    if (meta.length) {
      const line = document.createElement("span");
      line.className = "meta";
      line.textContent = meta.join(" · ");
      li.append(line);
    }
    if (item.citation) {
      const source = document.createElement("span");
      source.className = "meta";
      source.textContent = item.citation;
      li.append(source);
    }
    citations.append(li);
  }
}

function renderDocuments(items) {
  const current = documentSelect.value;
  documentSelect.replaceChildren();
  addOption("", "All documents");
  for (const item of items) {
    if (!item || typeof item.id !== "string") continue;
    addOption(item.id, item.path || item.name || item.id);
  }
  const stillThere = [...documentSelect.options].some((option) => option.value === current);
  if (stillThere) documentSelect.value = current;
}

async function loadDocuments() {
  try {
    const response = await fetch("/api/documents");
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      if (askAllowed) showError(errorMessage(payload, response.status));
      return;
    }
    const items = payload && Array.isArray(payload.documents) ? payload.documents : [];
    renderDocuments(items);
  } catch (error) {
    showError("Could not load the document list.");
  }
}

function knowledgeText(payload) {
  const count = payload.documents + " documents, " + payload.units + " search units";
  if (payload.state === "processing") {
    return "Processing — " + (payload.stage || "indexing") + ". " + count + ".";
  }
  if (payload.state === "ready") return "Ready. " + count + ".";
  if (payload.state === "failed" && payload.previous_available) {
    return "Indexing failed. The previous documents are still searchable. " + (payload.error || "");
  }
  if (payload.state === "failed") return "Indexing failed. " + (payload.error || "");
  return payload.error || "No index is available.";
}

function applyStatus(payload) {
  knowledge.className = "knowledge " + (payload.state || "");
  knowledge.textContent = knowledgeText(payload);
  askAllowed = Boolean(payload.previous_available) && payload.state !== "processing";
  const busy = payload.state === "processing";
  addFiles.disabled = busy;
  addFolder.disabled = busy;
  uploadButton.disabled = busy || pending.length === 0;
  refreshAskButton();
  if (busy && !pollTimer) pollTimer = window.setInterval(refreshStatus, 1000);
  if (!busy && pollTimer) {
    window.clearInterval(pollTimer);
    pollTimer = 0;
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/ingestion/status");
    const payload = await response.json().catch(() => null);
    if (!response.ok || !payload) return;
    const wasProcessing = knowledge.classList.contains("processing");
    applyStatus(payload);
    if (wasProcessing && payload.state === "ready") loadDocuments();
  } catch (error) {
    knowledge.className = "knowledge failed";
    knowledge.textContent = "Could not read the indexing status.";
  }
}

function relativeName(file) {
  return (file.webkitRelativePath || file.name || "").replaceAll("\\", "/");
}

function rememberFiles(fileListInput) {
  const chosen = [...fileListInput];
  const accepted = [];
  let skipped = 0;
  for (const file of chosen) {
    const name = relativeName(file).toLowerCase();
    if (SUPPORTED.some((suffix) => name.endsWith(suffix))) accepted.push(file);
    else skipped += 1;
  }
  pending = accepted;
  fileList.replaceChildren();
  for (const file of accepted) {
    const item = document.createElement("li");
    item.textContent = relativeName(file);
    fileList.append(item);
  }
  selection.hidden = false;
  selection.textContent = accepted.length
    ? accepted.length + (accepted.length === 1 ? " file selected." : " files selected.")
    : "No PDF or DOCX files selected.";
  if (skipped) selection.textContent += " Skipped " + skipped + " unsupported file" + (skipped === 1 ? "." : "s.");
  uploadButton.hidden = accepted.length === 0;
  uploadButton.disabled = accepted.length === 0;
}

function uploadFiles() {
  if (!pending.length) return;
  clearError();
  const body = new FormData();
  for (const file of pending) body.append("files", file, relativeName(file));
  addFiles.disabled = true;
  addFolder.disabled = true;
  uploadButton.disabled = true;
  selection.textContent = "Uploading " + pending.length + (pending.length === 1 ? " file…" : " files…");
  const request = new XMLHttpRequest();
  request.open("POST", "/api/documents");
  request.upload.addEventListener("progress", (event) => {
    if (!event.lengthComputable) return;
    selection.textContent = "Uploading " + Math.round((100 * event.loaded) / event.total) + "%";
  });
  request.addEventListener("load", () => {
    const payload = parseJson(request.responseText);
    if (request.status >= 200 && request.status < 300) {
      pending = [];
      fileList.replaceChildren();
      uploadButton.hidden = true;
      selection.textContent = "Upload received. Indexing has started.";
      if (payload && payload.status) applyStatus(payload.status);
      refreshStatus();
      return;
    }
    showError(errorMessage(payload, request.status));
    selection.textContent = "";
    addFiles.disabled = false;
    addFolder.disabled = false;
    uploadButton.disabled = false;
  });
  request.addEventListener("error", () => {
    showError("Could not reach the server.");
    addFiles.disabled = false;
    addFolder.disabled = false;
    uploadButton.disabled = false;
  });
  request.send(body);
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch (error) {
    return null;
  }
}

async function ask(event) {
  event.preventDefault();
  clearError();
  result.hidden = true;
  const question = questionInput.value.trim();
  if (!question) {
    showError("Enter a question.");
    return;
  }
  if (!askAllowed) {
    showError(knowledge.textContent || "Questions are available when the knowledge base is Ready.");
    return;
  }
  asking = true;
  status.hidden = false;
  refreshAskButton();
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        document: documentSelect.value || null,
      }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showError(errorMessage(payload, response.status));
      return;
    }
    answer.textContent = payload && typeof payload.answer === "string" ? payload.answer : "";
    renderCitations(payload && payload.citations);
    result.hidden = false;
  } catch (error) {
    showError("Could not reach the server.");
  } finally {
    asking = false;
    status.hidden = true;
    refreshAskButton();
  }
}

addFiles.addEventListener("click", () => fileInput.click());
addFolder.addEventListener("click", () => folderInput.click());
fileInput.addEventListener("change", () => {
  rememberFiles(fileInput.files || []);
  fileInput.value = "";
});
folderInput.addEventListener("change", () => {
  rememberFiles(folderInput.files || []);
  folderInput.value = "";
});
uploadButton.addEventListener("click", uploadFiles);
form.addEventListener("submit", ask);
refreshStatus();
loadDocuments();
