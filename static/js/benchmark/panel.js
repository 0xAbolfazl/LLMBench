// Setup station controller.
//
// One set of inputs decides everything: the prompts, the models ticked, and
// the configurations built. What kind of test that adds up to follows from the
// counts rather than from a mode the user has to pick first —
//
//   one model  · no or one configuration  → measure that model
//   one model  · several configurations   → compare the configurations
//   many models · no or one configuration → compare the models
//   many models · several configurations  → tournament: each model races
//                                           under the configuration paired
//                                           with it
//
// The launch console names the kind before the run starts; the run button
// posts to the single run endpoint with the shape that kind belongs to. While
// a run is in flight the run dock carries the progress; the finished results
// land back in this station.

import { api, postJson } from "../lib/api.js";
import { escapeHtml, hideAlert, showAlert } from "../lib/format.js";
import { getInstalledModels, getOllamaStatus, getModelCatalog } from "../lib/state.js";
import {
  fetchEngines,
  getCurrentEngine,
  getCurrentEngineEntry,
  getEngines,
  onEnginesChange,
  setCurrentEngine,
} from "../lib/engines.js";
import { fetchModels, fetchServerStatus, isModelLoaded, onServerChange } from "../lib/server.js";
import { setButtonBusy, toast } from "../lib/toast.js";
import { openModal } from "../lib/modal.js";
import { hideDock, initRunDock, paintDock, showDock } from "../run-dock.js";
import {
  addConfiguration,
  addConfigurations,
  clearConfigurations,
  duplicateConfiguration,
  getConfiguration,
  getConfigurations,
  removeConfiguration,
  toPayload,
  updateConfiguration,
} from "./store.js";
import { openEditor, setSchema } from "./editor.js";
import { renderConfigList } from "./configs.js";
import { renderResults } from "./results.js";
import { openTab } from "../tabs.js";

// A single prompt can take a minute on a large model, so polling is
// deliberately slow; the dock's clock interpolates between polls.
const POLL_INTERVAL_MS = 1500;

const MAX_MODELS = 6;

const ENDPOINTS = {
  run: "/api/benchmark/run",
  status: "/api/benchmark/status",
  clear: "/api/benchmark/clear",
};

const errorEl = document.getElementById("bench-error");
const metaEl = document.getElementById("run-meta");
const promptsEl = document.getElementById("bench-prompts");
const repetitionsEl = document.getElementById("bench-repetitions");
const includeOutputEl = document.getElementById("bench-include-output");

const engineSelectEl = document.getElementById("engine-select");
const modelListEl = document.getElementById("model-list");
const modelCountEl = document.getElementById("model-count");
const modelHintEl = document.getElementById("model-hint");

const configListEl = document.getElementById("config-list");
const configCountEl = document.getElementById("config-count");
const addBtn = document.getElementById("btn-config-add");
const importBtn = document.getElementById("btn-config-import");
const clearBtn = document.getElementById("btn-config-clear");
const promptCountEl = document.getElementById("prompt-count");
const pairingEl = document.getElementById("pairing");

const planEl = document.getElementById("plan");
const statsEl = document.getElementById("plan-stats");
const runBtn = document.getElementById("btn-run");
const runBtnLabel = document.getElementById("btn-run-label");
const readinessStateEl = document.getElementById("readiness-state");
const readinessNoteEl = document.getElementById("readiness-note");

const resultsEl = document.getElementById("results-body");
const resultsMetaEl = document.getElementById("results-meta");

// Models ticked, in tick order: a comparison measures them in this sequence.
let selectedModels = [];

// Tournament pairing: model name → configuration id. Only consulted when the
// plan is a tournament; everything else runs one shared configuration.
let pairing = new Map();

// The last parsed-but-not-yet-applied import.
let pendingImport = null;

let pollTimer = null;
let isRunning = false;

/**
 * Read the prompt textarea into a list.
 *
 * Prompts are separated by a blank line so a single prompt can still span
 * several lines, which is what a realistic test prompt usually does.
 *
 * @returns {string[]} Non-empty prompts.
 */
function readPrompts() {
  return promptsEl.value
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean);
}

/** Read the repetitions field, clamped to the range the server accepts. */
function readRepetitions() {
  const value = Number.parseInt(repetitionsEl.value, 10);
  return Number.isInteger(value) && value >= 1 ? Math.min(value, 10) : 1;
}

/**
 * Work out which of the four test kinds the current inputs describe.
 *
 * @returns {{kind: string, models: number, configs: number, prompts: number, reps: number, runs: number}}
 */
function readPlan() {
  const models = selectedModels.length;
  const configs = getConfigurations().length;
  const prompts = readPrompts().length;
  const reps = readRepetitions();

  let kind = "none";

  if (models === 1) {
    kind = configs > 1 ? "configs" : "single";
  } else if (models > 1) {
    kind = configs > 1 ? "tournament" : "models";
  }

  // A configs run varies the configurations; every other kind runs one
  // configuration (or the single default) per model.
  const passes = kind === "configs" ? configs : models;

  return { kind, models, configs, prompts, reps, runs: passes * prompts * reps };
}

/**
 * Paint the engine dropdown from the registry.
 */
function populateEngineSelect() {
  if (!engineSelectEl) {
    return;
  }

  const engines = getEngines();

  engineSelectEl.innerHTML = engines
    .map(
      (engine) =>
        `<option value="${escapeHtml(engine.id)}"${
          engine.id === getCurrentEngine() ? " selected" : ""
        }>${escapeHtml(engine.label)}</option>`
    )
    .join("");
}

/**
 * Bind the engine dropdown: selecting an engine moves the shared selection;
 * the onEnginesChange subscription below repaints everything engine-scoped.
 */
function bindEngineSelect() {
  if (!engineSelectEl) {
    return;
  }

  engineSelectEl.addEventListener("change", () => {
    setCurrentEngine(engineSelectEl.value);
  });
}

/**
 * Paint the model tick-list from the installed catalog.
 */
function renderModelList() {
  const catalog = getModelCatalog();
  const status = getOllamaStatus();

  const engineEntry = getCurrentEngineEntry();
  const engineLabel = engineEntry?.label || "The engine";

  if (catalog.length === 0) {
    let message = "No models installed.";

    if (engineEntry?.requires_server === false) {
      // Its model list comes from scanning folders, not a server.
      message = "No .gguf files found in the configured folders. Set the model folders in the engine station, then refresh.";
    } else if (status && status.running === false) {
      message = `The ${engineLabel} server is not running, so no models can be listed. Start it from the engine station.`;
    } else if (status && status.running) {
      message = `The ${engineLabel} server is running but reports no models. Add models to its library, then refresh.`;
    }

    modelListEl.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
    modelHintEl.textContent = "One model measures that model. Two or more race against each other, one at a time so they never share the GPU.";
    return;
  }

  modelListEl.innerHTML = catalog
    .map((model) => {
      const loaded = isModelLoaded(model.name);
      const size =
        model.size !== null && model.size !== undefined ? humanSize(model.size) : "";

      return `
        <div class="model-row${selectedModels.includes(model.name) ? " is-selected" : ""}"
             data-model-row="${escapeHtml(model.name)}" role="checkbox"
             aria-checked="${selectedModels.includes(model.name)}" tabindex="0">
          <span class="model-check" aria-hidden="true"></span>
          <span class="model-name" dir="ltr" title="${escapeHtml(model.name)}">${escapeHtml(model.name)}</span>
          <span class="model-meta">
            ${loaded ? `<span class="model-loaded">in vram</span>` : ""}
            ${size ? `<span>${escapeHtml(size)}</span>` : ""}
          </span>
        </div>`;
    })
    .join("");

  modelListEl.querySelectorAll("[data-model-row]").forEach((row) => {
    const toggle = () => toggleModel(row.dataset.modelRow);
    row.addEventListener("click", toggle);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggle();
      }
    });
  });

  modelHintEl.textContent =
    "One model measures that model. Two or more race against each other, one at a time so they never share the GPU.";
}

/**
 * A compact size for a model row: Ollama reports bytes.
 *
 * @param {number} bytes
 * @returns {string}
 */
function humanSize(bytes) {
  if (bytes >= 1024 ** 3) {
    return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  }
  if (bytes >= 1024 ** 2) {
    return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  }
  return `${(bytes / 1024).toFixed(0)} KB`;
}

/**
 * Toggle a model in or out of the selection.
 *
 * @param {string} name Model name.
 */
function toggleModel(name) {
  if (isRunning) {
    return;
  }

  if (selectedModels.includes(name)) {
    selectedModels = selectedModels.filter((model) => model !== name);
    pairing.delete(name);
  } else {
    if (selectedModels.length >= MAX_MODELS) {
      toast("error", "Too many models", `At most ${MAX_MODELS} models can be benchmarked in one run.`);
      return;
    }
    // Tick order is run order, so a newly ticked model goes last.
    selectedModels.push(name);
  }

  renderSetup();
}

/**
 * Paint the tournament's pairing table, or hide it when the plan is not one.
 */
function renderPairing() {
  const plan = readPlan();
  const configurations = getConfigurations();

  if (plan.kind !== "tournament") {
    pairingEl.classList.add("is-hidden");
    pairingEl.innerHTML = "";
    return;
  }

  pairingEl.classList.remove("is-hidden");

  const rows = selectedModels
    .map((model, index) => {
      // Unpaired models fall to the configuration at their own position, so a
      // tournament is runnable the moment it is described.
      const fallback = configurations[index % configurations.length];
      const chosen = pairing.get(model) ?? fallback.id;

      const options = configurations
        .map(
          (config) =>
            `<option value="${config.id}" ${config.id === chosen ? "selected" : ""}>${escapeHtml(
              config.name
            )}</option>`
        )
        .join("");

      return `
        <div class="pairing-row">
          <span class="pairing-model" dir="ltr">${escapeHtml(model)}</span>
          <select class="input pairing-select" dir="ltr" data-pair-model="${escapeHtml(model)}" ${
            isRunning ? "disabled" : ""
          }>
            ${options}
          </select>
        </div>`;
    })
    .join("");

  pairingEl.innerHTML = `
    <div class="pairing-title">Configuration per model</div>
    <div class="pairing-note">
      Several models and several configurations: give each model the one it
      races under.
    </div>
    ${rows}`;

  pairingEl.querySelectorAll("[data-pair-model]").forEach((select) => {
    select.addEventListener("change", () => {
      pairing.set(select.dataset.pairModel, Number(select.value));
      renderPlan();
    });
  });
}

/** Paint the plan panel: which of the four kinds the inputs describe. */
function renderPlan() {
  const plan = readPlan();

  const labels = {
    none: "Nothing to run",
    single: "Model speed test",
    configs: "Configuration test",
    models: "Model comparison",
    tournament: "Tournament",
  };

  if (plan.kind === "none") {
    planEl.className = "plan is-empty";
    planEl.innerHTML = `
      <span class="plan-kind">${escapeHtml(labels.none)}</span>
      <span class="plan-note">Pick at least one model and write at least one prompt.</span>`;
    return;
  }

  // The factors behind the total, in run order. Configurations only appear
  // when the plan actually varies them; the repetition count only earns its
  // place when it is above one.
  const factors = [];

  factors.push(`${plan.models} model${plan.models > 1 ? "s" : ""}`);

  if (plan.kind === "configs") {
    factors.push(`${plan.configs} config${plan.configs > 1 ? "s" : ""}`);
  }

  factors.push(`${plan.prompts} prompt${plan.prompts > 1 ? "s" : ""}`);

  if (plan.reps > 1) {
    factors.push(`${plan.reps}x reps`);
  }

  const counts = `<span class="plan-counts">${escapeHtml(
    factors.join(" × ")
  )} = ${plan.runs} run${plan.runs > 1 ? "s" : ""}</span>`;

  planEl.className = `plan is-${plan.kind}`;
  planEl.innerHTML = `
    <span class="plan-kind">${escapeHtml(labels[plan.kind])}</span>
    ${counts}`;
}

/**
 * Paint the four summary tiles: model count, configuration count, total
 * generations and the server state.
 */
function renderStats() {
  const plan = readPlan();
  const status = getOllamaStatus();
  const autoServer = getCurrentEngineEntry()?.requires_server === false;

  // An engine that spawns its server per run is never "down": it starts
  // from idle, so the tile reports that rather than a red "down".
  const serverTile = autoServer
    ? { value: "auto", tone: "" }
    : {
        value: status ? (status.running ? "up" : "down") : "—",
        tone: status ? (status.running ? "is-good" : "is-bad") : "is-unknown",
      };

  const stats = [
    { label: "Models", value: plan.models || "—", tone: plan.models ? "is-good" : "is-unknown" },
    { label: "Configs", value: plan.configs || "—", tone: "" },
    { label: "Runs", value: plan.runs || "—", tone: plan.runs ? "is-strong" : "is-unknown" },
    { label: "Server", ...serverTile },
  ];

  statsEl.innerHTML = stats
    .map(
      (stat) => `
        <div class="plan-stat">
          <span class="plan-stat-value ${stat.tone}" dir="auto">${escapeHtml(String(stat.value))}</span>
          <span class="plan-stat-label">${escapeHtml(stat.label)}</span>
        </div>`
    )
    .join("");
}

/** Refresh everything that depends on the current setup. */
function renderSetup() {
  const configurations = getConfigurations();
  const prompts = readPrompts();

  renderModelList();
  renderConfigList(configListEl);
  renderPairing();
  renderPlan();
  renderStats();

  modelCountEl.textContent =
    selectedModels.length === 0
      ? "no model selected"
      : `${selectedModels.length} selected`;

  promptCountEl.textContent =
    prompts.length === 0 ? "no prompts written" : `${prompts.length} prompt${prompts.length > 1 ? "s" : ""}`;

  configCountEl.textContent =
    configurations.length === 0
      ? "none built"
      : `${configurations.length} built`;

  clearBtn.disabled = configurations.length === 0 || isRunning;
  repetitionsEl.disabled = isRunning;

  updateReadiness();
}

/** Enable or disable the run button and explain whichever state it is in. */
function updateReadiness() {
  if (isRunning) {
    readinessStateEl.className = "readiness-state";
    readinessStateEl.textContent = "in progress";
    readinessNoteEl.textContent = "A benchmark is running. Wait for it to finish.";
    runBtn.disabled = true;
    runBtn.classList.add("is-running");
    runBtnLabel.textContent = "Running…";
    return;
  }

  const plan = readPlan();
  const status = getOllamaStatus();
  const engine = getCurrentEngineEntry();
  const blockers = [];

  // Only an engine with a resident server must be up before a run. An engine
  // that spawns and stops a server per model during the run (llama.cpp) is
  // fine to start from its normal "nothing running" idle state.
  if (status && status.running === false && engine?.requires_server !== false) {
    blockers.push(`${engine?.label || "the engine"} is not running`);
  }

  if (plan.models === 0) {
    blockers.push("no model selected");
  }

  if (plan.prompts === 0) {
    blockers.push("no prompts written");
  }

  runBtn.classList.remove("is-running");
  runBtnLabel.textContent = "Run benchmark";

  if (blockers.length > 0) {
    readinessStateEl.className = "readiness-state is-bad";
    readinessStateEl.textContent = "not ready";
    readinessNoteEl.textContent = `Still missing: ${blockers.join(", ")}.`;
    runBtn.disabled = true;
    return;
  }

  readinessStateEl.className = "readiness-state is-good";
  readinessStateEl.textContent = "ready";
  readinessNoteEl.textContent = `Ready to run ${plan.runs} generation${plan.runs > 1 ? "s" : ""}.`;
  runBtn.disabled = false;
}

/* ------------------------------------------------------------------ *
 * Configuration import (modal)
 * ------------------------------------------------------------------ */

/**
 * Build and open the import modal.
 */
function openImport() {
  const modal = openModal(
    "Import configurations",
    `
      <div class="dropzone" id="import-dropzone" tabindex="0" role="button">
        <span class="dropzone-glyph" aria-hidden="true">⬒</span>
        <span class="dropzone-text">Drop a JSON file here, or click to browse</span>
        <span class="dropzone-hint">A file holding one or more configurations.</span>
        <input type="file" id="import-file" accept=".json,application/json" hidden>
      </div>
      <div class="paste-box">
        <label class="field">
          <span class="field-label">…or paste the configuration</span>
          <textarea class="input mono" id="import-paste-text" rows="7" dir="ltr" spellcheck="false"
            placeholder='{
  "configurations": [
    { "name": "baseline", "options": { "temperature": 0.7, "num_ctx": 4096 } },
    { "name": "long-context", "options": { "temperature": 0.7, "num_ctx": 16384 } }
  ]
}'></textarea>
        </label>
        <div class="btn-row" style="justify-content:flex-end">
          <button type="button" class="btn btn-sm" id="btn-import-check">Check</button>
        </div>
      </div>
      <div class="import-preview is-hidden" id="import-preview" aria-live="polite"></div>`
  );

  const dropzone = modal.el.querySelector("#import-dropzone");
  const fileEl = modal.el.querySelector("#import-file");
  const pasteEl = modal.el.querySelector("#import-paste-text");
  const checkBtn = modal.el.querySelector("#btn-import-check");
  const previewEl = modal.el.querySelector("#import-preview");

  dropzone.addEventListener("click", () => fileEl.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileEl.click();
    }
  });

  fileEl.addEventListener("change", () => {
    checkFile(fileEl.files[0]);
    // Reset so choosing the same file twice in a row fires change again.
    fileEl.value = "";
  });

  ["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-hover");
    });
  });

  ["dragleave", "drop"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-hover");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    checkFile(event.dataTransfer.files[0]);
  });

  checkBtn.addEventListener("click", () => checkPastedText());

  function clearPreview() {
    pendingImport = null;
    previewEl.classList.add("is-hidden", "is-bad");
    previewEl.innerHTML = "";
  }

  /**
   * Show why an import could not be read.
   *
   * @param {string} message Server or client message.
   */
  function showPreviewError(message) {
    pendingImport = null;
    previewEl.classList.remove("is-hidden");
    previewEl.classList.add("is-bad");
    previewEl.innerHTML = `
      <div class="preview-head">
        <span class="preview-title">Could not read the file</span>
      </div>
      <div class="preview-error">${escapeHtml(message)}</div>`;
  }

  /**
   * Show what an import was understood to contain, before applying it.
   *
   * Applying is a separate step because a file may also carry a model and
   * prompts, which would otherwise silently overwrite what the user already
   * typed.
   *
   * @param {object} document_ Parsed document from the parse endpoint.
   * @param {boolean} [fromFile] Whether the source is a file name rather than
   *     the paste box.
   */
  function showPreview(document_, fromFile = true) {
    pendingImport = document_;

    const source = fromFile ? document_.source || "" : "pasted text";

    const rows = document_.configurations
      .map(
        (config) => `
          <div class="preview-row">
            <span class="preview-name">${escapeHtml(config.name)}</span>
            <span class="preview-opts">${escapeHtml(
              Object.entries(config.options)
                .map(([key, value]) => `${key}=${Array.isArray(value) ? value.join("|") : value}`)
                .join("  ")
            )}</span>
          </div>`
      )
      .join("");

    const extras = [];

    if (document_.model) {
      extras.push(`model “${document_.model}”`);
    }

    if (document_.prompts.length > 0) {
      extras.push(`${document_.prompts.length} prompt${document_.prompts.length > 1 ? "s" : ""}`);
    }

    const note = extras.length
      ? `<span class="preview-source">Also carries ${escapeHtml(extras.join(" and "))}.</span>`
      : "";

    previewEl.classList.remove("is-hidden", "is-bad");
    previewEl.innerHTML = `
      <div class="preview-head">
        <span class="preview-title">Found ${document_.configurations.length} configuration${
          document_.configurations.length > 1 ? "s" : ""
        }</span>
        <span class="preview-source">${escapeHtml(source)}</span>
        ${note}
      </div>
      <div class="preview-list">${rows}</div>
      <div class="preview-foot">
        <button type="button" class="btn btn-sm" data-import-act="replace">Replace existing</button>
        <button type="button" class="btn btn-sm btn-accent" data-import-act="append">Add to list</button>
      </div>`;

    previewEl.querySelectorAll("[data-import-act]").forEach((button) => {
      button.addEventListener("click", () => {
        applyImport(button.dataset.importAct);
        modal.close();
      });
    });
  }

  /**
   * Validate an uploaded file against the server's schema.
   *
   * @param {File} file File chosen or dropped by the user.
   */
  async function checkFile(file) {
    if (!file) {
      return;
    }

    const form = new FormData();
    form.append("file", file);

    try {
      const { data } = await api("/api/benchmark/parse", { method: "POST", body: form });
      showPreview(data);
    } catch (error) {
      showPreviewError(error.message);
    }
  }

  /** Validate whatever is in the paste box. */
  async function checkPastedText() {
    const text = pasteEl.value.trim();

    if (!text) {
      showPreviewError("Paste a configuration first.");
      return;
    }

    try {
      const { data } = await postJson("/api/benchmark/parse", {
        engine: getCurrentEngine(),
        text,
      });
      showPreview(data, false);
    } catch (error) {
      showPreviewError(error.message);
    }
  }
}

/**
 * Apply a previewed import.
 *
 * @param {"append"|"replace"} mode Whether to keep the existing configurations.
 */
function applyImport(mode) {
  if (!pendingImport) {
    return;
  }

  if (mode === "replace") {
    clearConfigurations();
    pairing.clear();
  }

  addConfigurations(pendingImport.configurations);

  // A file may describe a whole setup, not just configurations. Those fields
  // are only taken when the user has not already chosen them themselves.
  if (
    pendingImport.model &&
    getInstalledModels().includes(pendingImport.model) &&
    selectedModels.length === 0
  ) {
    selectedModels = [pendingImport.model];
  }

  if (pendingImport.prompts.length > 0 && (mode === "replace" || readPrompts().length === 0)) {
    promptsEl.value = pendingImport.prompts.join("\n\n");
  }

  if (pendingImport.repetitions && readRepetitions() === 1) {
    repetitionsEl.value = pendingImport.repetitions;
  }

  if (pendingImport.include_output !== null && pendingImport.include_output !== undefined) {
    includeOutputEl.checked = pendingImport.include_output;
  }

  toast(
    "success",
    mode === "replace" ? "Configurations replaced" : "Configurations added",
    `${getConfigurations().length} in the comparison now.`
  );

  renderSetup();
}

/* ------------------------------------------------------------------ *
 * Running a benchmark
 * ------------------------------------------------------------------ */

/**
 * Lock or unlock every control that would change the setup mid-run.
 *
 * @param {boolean} running Whether a benchmark is in flight.
 */
function setRunning(running) {
  isRunning = running;

  [promptsEl, includeOutputEl, repetitionsEl, addBtn, importBtn].forEach(
    (el) => {
      el.disabled = running;
    }
  );

  renderSetup();
}

/**
 * Download the finished job's result through one of the results endpoints.
 *
 * @param {string} format "csv" or "json".
 * @param {object} job The finished job snapshot.
 */
async function downloadResult(format, job) {
  const response = await fetch(
    format === "csv" ? "/api/benchmark/results-csv" : "/api/benchmark/results-json",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: job.id, result: job.result }),
    }
  );

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `Request failed (${response.status})`);
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = `benchmark-${job.id}.${format}`;
  link.click();
  URL.revokeObjectURL(url);
}

/**
 * Apply a job snapshot to the page.
 *
 * @param {object|null} job Snapshot from the run or status endpoint.
 */
function applyJob(job) {
  if (!job) {
    setRunning(false);
    hideDock();
    metaEl.textContent = "";
    return;
  }

  if (job.status === "running") {
    setRunning(true);
    showDock(job);
    paintDock(job);
    metaEl.textContent = `running since ${job.started_at}`;
    return;
  }

  setRunning(false);
  hideDock();

  if (job.status === "failed") {
    showAlert(errorEl, `The run failed: ${job.error}`);
    resultsEl.innerHTML = `<div class="empty-state">The run did not finish.</div>`;
    resultsMetaEl.textContent = `failed at ${job.finished_at || ""}`;
    return;
  }

  if (job.status === "cancelled") {
    hideAlert(errorEl);
    resultsEl.innerHTML = `<div class="empty-state">The run was cancelled; partial results were discarded.</div>`;
    resultsMetaEl.textContent = `cancelled at ${job.finished_at || ""}`;
    return;
  }

  hideAlert(errorEl);
  renderResults(resultsEl, job);
  resultsMetaEl.textContent = `finished at ${job.finished_at || ""}`;

  const discard = resultsEl.querySelector("[data-results-discard]");

  if (discard) {
    discard.addEventListener("click", async () => {
      try {
        await postJson(ENDPOINTS.clear, {});
      } catch (error) {
        toast("error", "Could not discard", error.message);
        return;
      }
      resultsEl.innerHTML = `<div class="empty-state">No benchmark has been run yet.</div>`;
      resultsMetaEl.textContent = "";
    });
  }

  resultsEl.querySelectorAll("[data-results-download]").forEach((button) => {
    button.addEventListener("click", async () => {
      const restore = setButtonBusy(button);

      try {
        await downloadResult(button.dataset.resultsDownload, job);
      } catch (error) {
        toast("error", "Export failed", error.message);
      } finally {
        restore();
      }
    });
  });
}

/** Poll the status endpoint while a run is in flight. */
function startPolling() {
  window.clearInterval(pollTimer);

  pollTimer = window.setInterval(async () => {
    let job;

    try {
      job = (await api(ENDPOINTS.status)).data;
    } catch (error) {
      // A failed poll says nothing about the run itself, so keep polling and
      // let the alert explain the gap.
      showAlert(errorEl, `Lost track of the run: ${error.message}`);
      return;
    }

    hideAlert(errorEl);
    applyJob(job);

    if (!job || job.status !== "running") {
      window.clearInterval(pollTimer);
      pollTimer = null;

      if (job && job.status === "done") {
        toast("success", "Benchmark finished", `${job.result.tests.length} result${
          job.result.tests.length > 1 ? "s" : ""
        } ready.`);
      } else if (job && job.status === "failed") {
        toast("error", "Benchmark failed", job.error);
      } else if (job && job.status === "cancelled") {
        toast("info", "Benchmark cancelled", "Partial results were discarded.");
      }
    }
  }, POLL_INTERVAL_MS);
}

/**
 * Build the request body for the plan the inputs describe. Every kind posts
 * to the single run endpoint; the body's shape (models, configurations,
 * model_configs) is what selects the kind.
 *
 * @param {object} plan The plan from readPlan.
 * @returns {object} Payload to post.
 */
function buildRequest(plan) {
  const prompts = readPrompts();
  const configurations = toPayload();
  const payload = {
    engine: getCurrentEngine(),
    models: selectedModels,
    prompts,
    include_output: includeOutputEl.checked,
    repetitions: readRepetitions(),
    configurations: [],
    model_configs: {},
  };

  if (plan.models === 1) {
    // One model: its own configurations, or none for the plain speed test.
    payload.configurations = configurations;
    return payload;
  }

  if (plan.kind === "tournament") {
    // Several models, each racing the configuration paired with it.
    const stored = getConfigurations();

    selectedModels.forEach((model, index) => {
      const fallback = stored[index % stored.length];
      const chosenId = pairing.get(model) ?? fallback.id;
      const chosen = stored.find((config) => config.id === chosenId) || fallback;

      payload.model_configs[model] = chosen.options;
    });
  } else if (configurations.length === 1) {
    // Several models under one shared configuration.
    payload.configurations = configurations;
  }

  return payload;
}

/** Download the current setup as a file that can be uploaded again later. */
async function exportSetup() {
  const response = await fetch("/api/benchmark/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildRequest(readPlan())),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `Request failed (${response.status})`);
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = "benchmark-configurations.json";
  link.click();
  URL.revokeObjectURL(url);
}

/* ------------------------------------------------------------------ *
 * Wiring
 * ------------------------------------------------------------------ */

function bindInputs() {
  promptsEl.addEventListener("input", () => {
    const count = readPrompts().length;

    promptCountEl.textContent =
      count === 0 ? "no prompts written" : `${count} prompt${count > 1 ? "s" : ""}`;

    renderPlan();
    renderStats();
    updateReadiness();
  });

  repetitionsEl.addEventListener("input", () => {
    renderPlan();
    updateReadiness();
  });

  includeOutputEl.addEventListener("change", () => {
    renderStats();
  });

  document.getElementById("btn-models-refresh").addEventListener("click", async () => {
    await fetchModels();
    renderSetup();
    toast("success", "Models refreshed");
  });
}

function bindConfigList() {
  // Cards are re-rendered on every change, so the container carries the handler.
  configListEl.addEventListener("click", (event) => {
    const button = event.target.closest("[data-config-act]");

    if (!button || isRunning) {
      return;
    }

    const id = Number(button.dataset.configId);

    if (button.dataset.configAct === "edit") {
      const config = getConfiguration(id);

      if (config) {
        openEditor("edit", config, {
          onSave: (draft) => {
            const stored = updateConfiguration(id, draft);
            renderSetup();
            if (stored) {
              toast("success", "Configuration saved", stored.name);
            }
          },
          onCancel: () => {},
        });
      }
      return;
    }

    if (button.dataset.configAct === "copy") {
      const copy = duplicateConfiguration(id);
      renderSetup();
      if (copy) {
        toast("success", "Configuration duplicated", copy.name);
      }
      return;
    }

    if (button.dataset.configAct === "remove") {
      removeConfiguration(id);
      // A pairing that named it is stale; the table falls back to position.
      [...pairing.entries()].forEach(([model, configId]) => {
        if (configId === id) {
          pairing.delete(model);
        }
      });
      renderSetup();
    }
  });

  clearBtn.addEventListener("click", () => {
    if (!window.confirm("Remove all configurations?")) {
      return;
    }
    clearConfigurations();
    pairing.clear();
    renderSetup();
  });

  addBtn.addEventListener("click", () => {
    openEditor("add", null, {
      onSave: (draft) => {
        if (Object.keys(draft.options).length === 0) {
          toast("error", "Nothing to save", "Set at least one option for this configuration.");
          return;
        }
        const stored = addConfiguration(draft);
        toast("success", "Configuration added", stored.name);
        renderSetup();
      },
      onCancel: () => {},
    });
  });

  importBtn.addEventListener("click", openImport);
}

function bindRun() {
  document.getElementById("btn-export-setup").addEventListener("click", async () => {
    const exportBtn = document.getElementById("btn-export-setup");
    const restore = setButtonBusy(exportBtn);

    try {
      await exportSetup();
      toast("success", "Setup exported", "Saved as benchmark-configurations.json.");
    } catch (error) {
      toast("error", "Export failed", error.message);
    } finally {
      restore();
    }
  });

  runBtn.addEventListener("click", async () => {
    const plan = readPlan();
    const payload = buildRequest(plan);
    const restore = setButtonBusy(runBtn);

    try {
      const { data } = await postJson(ENDPOINTS.run, payload);

      hideAlert(errorEl);
      applyJob(data);
      startPolling();

      toast(
        "pending",
        "Benchmark started",
        `${data.planned_runs} generation${data.planned_runs > 1 ? "s" : ""} planned.`
      );
    } catch (error) {
      showAlert(errorEl, `Could not start: ${error.message}`);
      toast("error", "Could not start", error.message);
    } finally {
      restore();
      updateReadiness();
    }
  });
}

/**
 * Bind the setup station and pick up any run already in flight.
 *
 * Called from the bootstrap after the schema and the first server read.
 */
export async function initBenchmarkPanel() {
  bindInputs();
  bindConfigList();
  bindRun();
  initRunDock();

  renderSetup();

  // Build the configuration editor from the schema the server publishes,
  // and repaint it whenever the engine — each carries its own option table —
  // or its selection changes.
  async function loadSchema() {
    try {
      const { data } = await api(
        `/api/benchmark/schema?engine=${encodeURIComponent(getCurrentEngine() || "")}`
      );
      setSchema(data.options);
    } catch (error) {
      showAlert(errorEl, `Could not load the option schema: ${error.message}`);
    }
  }

  bindEngineSelect();

  onEnginesChange(() => {
    selectedModels = [];
    pairing.clear();
    renderSetup();
    loadSchema();
  });

  await fetchEngines();
  populateEngineSelect();
  await loadSchema();

  // The model list follows the shared catalog, so a refresh from the Ollama
  // station re-renders it here too.
  onServerChange((moved) => {
    if (moved.models) {
      renderModelList();
      renderStats();
      updateReadiness();
    }

    if (moved.status) {
      renderStats();
      updateReadiness();
    }
  });

  // A reload during a run must not lose it: ask the status endpoint and resume
  // polling if a run is in flight, otherwise show the last finished job.
  try {
    const { data } = await api(ENDPOINTS.status);

    if (data && data.status === "running") {
      applyJob(data);
      startPolling();
      return;
    }

    if (data && data.status !== undefined && data.status !== null) {
      applyJob(data);
    }
  } catch (error) {
    showAlert(errorEl, `Could not load the run status: ${error.message}`);
  }
}

/**
 * Jump to this station. Used by the Ollama station's "bench it" buttons.
 *
 * @param {string} [modelName] Model to tick while there.
 */
export function goToSetup(modelName) {
  openTab("setup");

  if (modelName && !selectedModels.includes(modelName)) {
    if (selectedModels.length < MAX_MODELS && !isRunning) {
      selectedModels.push(modelName);
      renderSetup();
    }
  }
}

