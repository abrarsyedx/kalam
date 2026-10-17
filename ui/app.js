// Kalam UI. Talks to the Rust core through Tauri commands and events.
"use strict";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
const el = {
  record: $("btn-record"), pause: $("btn-pause"), stop: $("btn-stop"),
  stateLabel: $("state-label"), timer: $("timer"),
  micMeter: $("meter-mic"), sysMeter: $("meter-sys"),
  paper: $("paper"), empty: $("empty"), lines: $("lines"), preview: $("preview"),
  folder: $("btn-folder"), saved: $("saved"), openDocx: $("btn-open-docx"), openMp3: $("btn-open-mp3"),
  modelChip: $("model-chip"), modelSelect: $("model-select"), modelState: $("model-state"),
  gpuBadge: $("gpu-badge"), polish: $("polish"), polishText: $("polish-text"),
  toast: $("toast"), dialog: $("settings"), form: $("settings-form"),
};

let mode = "idle"; // idle | starting | recording | paused | stopping
let settings = null;
let status = null;
let lastSaved = null;

// ------------------------------------------------------------ helpers ----

function clock(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds));
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor(s / 60) % 60)}:${pad(s % 60)}`;
}

let toastTimer = 0;
function toast(message, kind = "info", ms = 5000) {
  el.toast.textContent = message;
  el.toast.className = `toast ${kind}`;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.toast.hidden = true), ms);
}
el.toast.addEventListener("click", () => (el.toast.hidden = true));

function setButton(btn, icon, label) {
  btn.querySelector("use").setAttribute("href", `#i-${icon}`);
  btn.querySelector("span").textContent = label;
}

function setMode(next) {
  mode = next;
  const active = next === "recording" || next === "paused";
  document.body.classList.toggle("recording", next === "recording");
  document.body.classList.toggle("paused", next === "paused");
  el.record.disabled = next !== "idle";
  el.pause.disabled = !active;
  el.stop.disabled = !active;
  el.modelSelect.disabled = next !== "idle";
  setButton(el.pause, next === "paused" ? "play" : "pause", next === "paused" ? "Resume" : "Pause");
  el.pause.title = next === "paused" ? "Resume recording and transcribing" : "Pause recording and transcribing";
  el.stateLabel.textContent = {
    idle: "Ready", starting: "Starting…", recording: "Recording", paused: "Paused", stopping: "Saving…",
  }[next];
  if (!active) setLevels({ mic: 0, system: 0 });
}

function setLevels({ mic, system }) {
  const width = (peak) => {
    if (!peak || peak <= 0) return 0;
    const db = 20 * Math.log10(peak);
    return Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  };
  for (const [meter, peak] of [[el.micMeter, mic], [el.sysMeter, system]]) {
    const fill = meter.querySelector(".fill");
    const w = width(peak);
    fill.style.width = `${w}%`;
    fill.classList.toggle("hot", w > 92);
  }
}

function updateSources(info) {
  const micOn = info ? !!info.micDevice : settings?.captureMic;
  const sysOn = info ? !!info.systemDevice : settings?.captureSystem;
  el.micMeter.classList.toggle("off", !micOn);
  el.sysMeter.classList.toggle("off", !sysOn);
  el.micMeter.title = info?.micDevice ? `Microphone: ${info.micDevice}` : "Microphone";
  el.sysMeter.title = info?.systemDevice ? `Computer audio: ${info.systemDevice}` : "Computer audio (Teams, Zoom, Meet…)";
}

function nearBottom() {
  const p = el.paper;
  return p.scrollHeight - p.scrollTop - p.clientHeight < 120;
}

function scrollToEnd(force) {
  if (force) el.paper.scrollTop = el.paper.scrollHeight;
}

function showWriting() {
  el.empty.hidden = true;
  el.paper.classList.add("writing");
}

/** Colour class for a speaker label: "You", or "Speaker N" (cycles 8 colours). */
function speakerClass(label) {
  if (label === "You") return " you";
  const n = /^Speaker (\d+)$/.exec(label || "");
  return n ? ` s${(Number(n[1]) - 1) % 8}` : "";
}

function addSegments(segments) {
  const stick = nearBottom();
  showWriting();
  for (const seg of segments) {
    const p = document.createElement("p");
    p.className = "line";
    const time = document.createElement("time");
    time.textContent = clock(seg.startMs / 1000);
    const who = document.createElement("span");
    who.className = "who" + speakerClass(seg.speaker);
    who.textContent = seg.speaker || "";
    const text = document.createElement("span");
    text.className = "text";
    text.textContent = seg.text;
    p.append(time, who, text);
    el.lines.append(p);
  }
  scrollToEnd(stick);
}

function setPreview(preview) {
  if (!preview || !preview.text) {
    el.preview.hidden = true;
    return;
  }
  const stick = nearBottom();
  showWriting();
  el.preview.querySelector(".text").textContent = preview.text;
  el.preview.hidden = false;
  scrollToEnd(stick);
}

function clearTranscript() {
  el.lines.replaceChildren();
  el.preview.hidden = true;
  el.empty.hidden = false;
  el.paper.classList.remove("writing");
}

const fileName = (p) => (p || "").split(/[\\/]/).pop();

/** Fill the header model switcher from the bundled models (+ a custom file). */
function renderModelSelect() {
  if (!status || !settings) return;
  const options = status.models
    .filter((m) => m.available || m.id === settings.model)
    .map((m) => [m.id, `${m.name} · ${m.detail}`]);
  if (settings.modelPath) options.push(["custom", `Custom · ${fileName(settings.modelPath)}`]);
  fillSelect(el.modelSelect, options, settings.model);
}

function setModelStatus(m) {
  if (!m) return;
  el.modelChip.classList.remove("loading", "ready", "failed");
  if (m.state === "loading") {
    el.modelChip.classList.add("loading");
    el.modelState.textContent = "loading…";
    el.modelChip.title = `Loading the speech model ${m.name}…`;
  } else if (m.state === "ready") {
    el.modelChip.classList.add("ready");
    el.modelState.textContent = "";
    el.modelChip.title = `Speech model ${m.name} is ready (loaded in ${(m.loadMs / 1000).toFixed(1)} s) and runs fully offline. Click to switch models.`;
  } else {
    el.modelChip.classList.add("failed");
    el.modelState.textContent = "failed";
    el.modelChip.title = m.error;
    toast(`${m.error}\nRecording still works; switch to another model.`, "error", 9000);
  }
  if (status) status.model = m;
}

/** Graphics card status: badge in the header and the line in Settings. */
function setGpuStatus(g) {
  if (!g) return;
  if (status) status.gpu = g;
  const line = $("gpu-status");
  const badge = el.gpuBadge;
  badge.hidden = g.state === "unsupported";
  badge.classList.toggle("cpu", g.state !== "active");
  line.classList.toggle("cpu", g.state !== "active");
  if (g.state === "active") {
    const faster = g.speedup ? ` (${g.speedup.toFixed(1)}× faster than the processor)` : "";
    badge.textContent = "GPU";
    badge.title = `Speech recognition runs on ${g.device}`;
    line.textContent = `Using ${g.device}${faster}.`;
  } else if (g.state === "checking") {
    badge.textContent = "…";
    badge.title = "Testing the graphics card…";
    line.textContent = "Testing the graphics card… (only on first start, about 30 seconds)";
  } else if (g.state === "disabled") {
    badge.textContent = "CPU";
    badge.title = "Graphics card turned off in Settings";
    line.textContent = "Turned off: speech recognition runs on the processor.";
  } else if (g.state === "cpu") {
    badge.textContent = "CPU";
    badge.title = g.reason;
    line.textContent = `Using the processor. ${g.reason}`;
  } else {
    line.textContent = "This version of Kalam runs on the processor only.";
  }
}

/** Polish pass progress in the footer. */
function setPolish(update) {
  if (!update || update.state !== "running") {
    el.polish.hidden = true;
    return;
  }
  el.polish.hidden = false;
  const pct = Math.round((update.progress || 0) * 100);
  const what = update.stage === "speakers" ? "Identifying speakers" : "Polishing transcript";
  el.polishText.textContent = `${what}… ${pct}%`;
  el.polish.querySelector(".fill").style.width = `${pct}%`;
}

$("btn-polish-skip").addEventListener("click", async () => {
  el.polishText.textContent = "Skipping…";
  try {
    await invoke("cancel_polish");
  } catch (e) {
    toast(String(e), "error");
  }
});

el.modelSelect.addEventListener("change", async () => {
  const choice = el.modelSelect.value;
  const label = el.modelSelect.selectedOptions[0]?.textContent || choice;
  try {
    settings = await invoke("select_model", { model: choice });
    toast(`Speech model: ${label}`);
  } catch (e) {
    toast(String(e), "error");
  }
  renderModelSelect();
});

function showFolder() {
  el.folder.textContent = settings?.outputDir || "";
  el.folder.title = `Open ${settings?.outputDir || "folder"}`;
}

// ----------------------------------------------------------- actions -----

async function start() {
  if (mode !== "idle") return;
  setMode("starting");
  try {
    const polishing = !el.polish.hidden;
    const info = await invoke("start_recording");
    clearTranscript();
    el.saved.hidden = true;
    setPolish(null);
    if (polishing) toast("Polishing of the previous transcript was stopped; its live version is saved.");
    el.timer.textContent = clock(0);
    updateSources(info);
    setMode("recording");
    if (info.warnings?.length) toast(info.warnings.join("\n"), "error", 9000);
    if (status?.model?.state === "loading") toast("Recording. Transcription will begin as soon as the speech model has loaded.");
  } catch (e) {
    setMode("idle");
    toast(String(e), "error", 9000);
  }
}

async function togglePause() {
  try {
    if (mode === "recording") {
      await invoke("pause_recording");
      setMode("paused");
    } else if (mode === "paused") {
      await invoke("resume_recording");
      setMode("recording");
    }
  } catch (e) {
    toast(String(e), "error");
  }
}

async function stop() {
  if (mode !== "recording" && mode !== "paused") return;
  setMode("stopping");
  try {
    const saved = await invoke("stop_recording");
    lastSaved = saved;
    el.saved.hidden = false;
    el.timer.textContent = clock(saved.durationSecs);
    const name = (p) => p.split(/[\\/]/).pop();
    el.openDocx.title = `Open ${name(saved.docxPath)}`;
    el.openMp3.title = `Open ${name(saved.mp3Path)}`;
    if (saved.warnings?.length) toast(saved.warnings.join("\n"), "error", 9000);
    else if (saved.polishing) toast(`Saved ${name(saved.mp3Path)} and ${name(saved.docxPath)}. Polishing the transcript now…`);
    else toast(`Saved ${name(saved.mp3Path)} and ${name(saved.docxPath)}`);
    if (saved.polishing) setPolish({ state: "running", progress: 0 });
  } catch (e) {
    toast(String(e), "error", 9000);
  }
  setPreview(null);
  setMode("idle");
}

async function openPath(path) {
  try {
    await invoke("open_path", { path });
  } catch (e) {
    toast(String(e), "error");
  }
}

el.record.addEventListener("click", start);
el.pause.addEventListener("click", togglePause);
el.stop.addEventListener("click", stop);
el.openDocx.addEventListener("click", () => lastSaved && openPath(lastSaved.docxPath));
el.openMp3.addEventListener("click", () => lastSaved && openPath(lastSaved.mp3Path));
el.folder.addEventListener("click", async () => {
  try {
    await invoke("reveal_path", { path: settings.outputDir });
  } catch (e) {
    toast(String(e), "error");
  }
});

// ----------------------------------------------------------- settings ----

// English first, then the EU languages Whisper supports, then a few others.
const LANGUAGES = [
  ["en", "English"], ["auto", "Detect automatically"],
  ["", "── European Union languages ──", true],
  ["bg", "Bulgarian"], ["hr", "Croatian"], ["cs", "Czech"], ["da", "Danish"], ["nl", "Dutch"],
  ["et", "Estonian"], ["fi", "Finnish"], ["fr", "French"], ["de", "German"], ["el", "Greek"],
  ["hu", "Hungarian"], ["it", "Italian"], ["lv", "Latvian"], ["lt", "Lithuanian"], ["mt", "Maltese"],
  ["pl", "Polish"], ["pt", "Portuguese"], ["ro", "Romanian"], ["sk", "Slovak"], ["sl", "Slovenian"],
  ["es", "Spanish"], ["sv", "Swedish"],
  ["", "── Other languages ──", true],
  ["ar", "Arabic"], ["zh", "Chinese"], ["hi", "Hindi"], ["ja", "Japanese"], ["ko", "Korean"],
  ["no", "Norwegian"], ["ru", "Russian"], ["tr", "Turkish"], ["uk", "Ukrainian"], ["ur", "Urdu"],
];

const f = (id) => $(id);
let draftModel = "accurate";
let draftModelPath = null;

function fillSelect(select, options, value) {
  select.replaceChildren(
    ...options.map(([v, label, separator]) => {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = label;
      o.disabled = !!separator;
      return o;
    })
  );
  select.value = value ?? "";
  if (select.value !== (value ?? "")) select.value = options[0]?.[0] ?? "";
}

function isEnglishOnly(choice, path) {
  if (choice === "custom") return /\.en([.\-_]|$)/i.test(fileName(path));
  return status?.models.find((m) => m.id === choice)?.englishOnly ?? false;
}

function refreshModelField() {
  document.querySelectorAll('input[name="f-model"]').forEach((r) => (r.checked = r.value === draftModel));
  f("f-model-path").textContent = draftModelPath || "Any whisper.cpp ggml-*.bin file";
  for (const m of status?.models || []) {
    const radio = document.querySelector(`input[name="f-model"][value="${m.id}"]`);
    if (radio) radio.disabled = !m.available;
  }
  const english = isEnglishOnly(draftModel, draftModelPath);
  f("f-language").disabled = english;
  f("lang-hint").textContent = english
    ? "This model understands English only. Choose Accurate to transcribe other languages."
    : "Pick the meeting's language for best results, or let Kalam detect it.";
  if (english) f("f-language").value = "en";
}

async function chooseModelFile() {
  try {
    const file = await invoke("pick_model_file");
    if (file) {
      draftModel = "custom";
      draftModelPath = file;
    }
  } catch (e) {
    toast(String(e), "error");
  }
  refreshModelField();
}

function refreshSourceFields() {
  f("f-mic-device").disabled = !f("f-mic").checked;
  f("f-sys-device").disabled = !f("f-sys").checked;
}

async function openSettings() {
  settings = await invoke("get_settings");
  draftModel = settings.model;
  draftModelPath = settings.modelPath || null;
  f("f-output").value = settings.outputDir;
  f("f-mic").checked = settings.captureMic;
  f("f-sys").checked = settings.captureSystem;
  f("f-preview").checked = settings.livePreview;
  f("f-vocab").value = settings.vocabulary || "";
  f("f-polish").value = settings.polish || "auto";
  f("f-gpu").value = settings.gpu || "auto";
  setGpuStatus(status?.gpu);
  f("f-speakers").value = settings.speakers || "voice";
  fillSelect(
    f("f-expected"),
    [["0", "Automatic"], ...Array.from({ length: 9 }, (_, i) => [String(i + 2), String(i + 2)])],
    String(settings.expectedSpeakers || 0)
  );
  refreshSpeakerFields();
  f("f-bitrate").value = String(settings.mp3Bitrate);
  fillSelect(f("f-language"), LANGUAGES, settings.language);
  const cores = navigator.hardwareConcurrency || 8;
  fillSelect(
    f("f-threads"),
    [["0", "Automatic (recommended)"], ...Array.from({ length: cores }, (_, i) => [String(i + 1), String(i + 1)])],
    String(settings.threads)
  );
  fillSelect(f("f-mic-device"), [["", "System default microphone"]], "");
  fillSelect(f("f-sys-device"), [["", "System default speakers / headset"]], "");
  f("sys-hint").textContent = navigator.userAgent.includes("Mac")
    ? "Requires macOS 14.6 or later. macOS asks once for permission to record system audio."
    : "Records whatever this device plays: use the speakers or headset your meeting app plays through.";
  refreshModelField();
  refreshSourceFields();
  f("about").textContent = `Kalam ${status?.version || ""} · open source (MIT) · speech recognition by Whisper / whisper.cpp, runs fully offline.`;
  el.dialog.showModal();

  try {
    const devices = await invoke("list_devices");
    const opts = (list, label) => [["", label], ...list.map((d) => [d.id, d.isDefault ? `${d.name} (default)` : d.name])];
    fillSelect(f("f-mic-device"), opts(devices.inputs, "System default microphone"), settings.micDevice || "");
    fillSelect(f("f-sys-device"), opts(devices.outputs, "System default speakers / headset"), settings.outputDevice || "");
    refreshSourceFields();
  } catch (e) {
    toast(`Could not list audio devices: ${e}`, "error");
  }
}

function refreshSpeakerFields() {
  f("f-expected-row").hidden = f("f-speakers").value !== "voice";
}

f("f-speakers").addEventListener("change", refreshSpeakerFields);
f("f-mic").addEventListener("change", refreshSourceFields);
f("f-sys").addEventListener("change", refreshSourceFields);
f("f-output-browse").addEventListener("click", async () => {
  try {
    const dir = await invoke("pick_folder", { current: f("f-output").value });
    if (dir) f("f-output").value = dir;
  } catch (e) {
    toast(String(e), "error");
  }
});
f("f-output-reset").addEventListener("click", () => (f("f-output").value = status?.defaultOutputDir || ""));
f("f-model-browse").addEventListener("click", chooseModelFile);
f("f-gpu-recheck").addEventListener("click", async () => {
  try {
    await invoke("recheck_gpu");
    setGpuStatus({ state: "checking" });
  } catch (e) {
    toast(String(e), "error");
  }
});
document.querySelectorAll('input[name="f-model"]').forEach((radio) =>
  radio.addEventListener("change", () => {
    if (radio.value === "custom" && !draftModelPath) return chooseModelFile();
    draftModel = radio.value;
    refreshModelField();
  })
);
el.dialog.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => el.dialog.close()));

el.form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const next = {
    outputDir: f("f-output").value.trim(),
    captureMic: f("f-mic").checked,
    captureSystem: f("f-sys").checked,
    micDevice: f("f-mic-device").value || null,
    outputDevice: f("f-sys-device").value || null,
    model: draftModel,
    modelPath: draftModelPath,
    language: f("f-language").value || "en",
    threads: Number(f("f-threads").value) || 0,
    mp3Bitrate: Number(f("f-bitrate").value) || 128,
    speakers: f("f-speakers").value,
    expectedSpeakers: Number(f("f-expected").value) || 0,
    livePreview: f("f-preview").checked,
    vocabulary: f("f-vocab").value.trim(),
    polish: f("f-polish").value,
    gpu: f("f-gpu").value,
  };
  if (!next.captureMic && !next.captureSystem) {
    toast("Turn on at least one audio source.", "error");
    return;
  }
  try {
    settings = await invoke("save_settings", { settings: next });
    showFolder();
    renderModelSelect();
    if (mode === "idle") updateSources(null);
    el.dialog.close();
    toast(mode === "idle" ? "Settings saved." : "Settings saved. Audio changes apply to the next recording.");
  } catch (e) {
    toast(String(e), "error", 8000);
  }
});

$("btn-settings").addEventListener("click", () => openSettings().catch((e) => toast(String(e), "error")));
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === ",") {
    e.preventDefault();
    if (!el.dialog.open) openSettings().catch((err) => toast(String(err), "error"));
  }
});

// -------------------------------------------------------------- boot -----

async function boot() {
  await Promise.all([
    listen("kalam://levels", (e) => {
      if (mode === "recording" || mode === "paused") {
        setLevels(e.payload);
        el.timer.textContent = clock(e.payload.seconds);
      }
    }),
    listen("kalam://preview", (e) => setPreview(e.payload)),
    listen("kalam://segments", (e) => {
      setPreview(null);
      addSegments(e.payload);
    }),
    listen("kalam://error", (e) => toast(String(e.payload), "error", 8000)),
    listen("kalam://model", (e) => setModelStatus(e.payload)),
    listen("kalam://gpu", (e) => setGpuStatus(e.payload)),
    listen("kalam://polish", (e) => {
      const u = e.payload;
      setPolish(u);
      if (u.state === "done") toast(u.message || "Transcript polished and saved.", u.message ? "error" : "info");
      if (u.state === "abandoned" && u.message) toast(u.message);
    }),
    listen("kalam://polished", (e) => {
      // Show the polished text, unless a new recording is already on screen.
      if (mode === "idle") {
        clearTranscript();
        addSegments(e.payload);
      }
    }),
  ]);

  [status, settings] = await Promise.all([invoke("get_status"), invoke("get_settings")]);
  setModelStatus(status.model);
  setGpuStatus(status.gpu);
  if (status.polishing) setPolish({ state: "running", progress: 0 });
  renderModelSelect();
  showFolder();
  if (status.recording) {
    updateSources(status.session);
    setMode(status.paused ? "paused" : "recording");
  } else {
    updateSources(null);
    setMode("idle");
  }
}

boot().catch((e) => toast(`Startup error: ${e}`, "error", 10000));
