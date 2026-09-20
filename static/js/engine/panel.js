// Engine station: the app-wide choice of which engine runs the models, and
// everything about it.
//
// The top is a bold chooser — one big card per engine — which owns the
// selection the whole app reads. Below it, the rest is scoped to whichever
// engine is chosen:
//
//   · configuration — where the engine lives and what it reads. llama.cpp is
//     configured here (its installation folder and the folders to scan for
//     .gguf models); Ollama reads its address from the environment and is
//     shown read-only.
//   · server — state, version and start/stop where the engine has a server.
//     llama.cpp has none to start on its own; the buttons adapt.
//   · models — what the engine can run: llama.cpp's recursive scan of the
//     configured folders, or Ollama's library.

import { api, postJson } from "../lib/api.js";
import { escapeHtml, humanBytes, humanDateTime } from "../lib/format.js";
import {
  getEngines,
  getCurrentEngine,
  getCurrentEngineEntry,
  setCurrentEngine,
  onEnginesChange,
} from "../lib/engines.js";
import {
  describeServer,
  fetchModels,
  refreshServer,
} from "../lib/server.js";
import {
  getLoadedModels,
  getModelCatalog,
  getOllamaStatus,
} from "../lib/state.js";
import { setButtonBusy, toast } from "../lib/toast.js";
import { onTabFirstOpen } from "../tabs.js";
import { goToSetup } from "../benchmark/panel.js";

const chooserEl = document.getElementById("engine-chooser");
const chooserHintEl = document.getElementById("engine-chooser-hint");
const configBodyEl = document.getElementById("engine-config-body");
const configMetaEl = document.getElementById("engine-config-meta");
const configApplyEl = document.getElementById("btn-engine-config-apply");
const metaEl = document.getElementById("engine-meta");
const cardsEl = document.getElementById("server-cards");
const stateLineEl = document.getElementById("server-state-line");
const startBtn = document.getElementById("btn-engine-start");
const stopBtn = document.getElementById("btn-engine-stop");
const modelCountEl = document.getElementById("engine-model-count");
const modelRowsEl = document.getElementById("engine-model-rows");
const modelsEmptyEl = document.getElementById("engine-models-empty");

// A short line under each engine card's name.
const ENGINE_NOTES = {
  ollama: "A resident server that keeps every pulled model ready on demand.",
  "llama-cpp":
    "One server process per model, spawned from the .gguf files you point it at.",
};

/** The current engine's id, or null. */
function engineId() {
  return getCurrentEngine();
}

/** The current engine's display label. */
function engineLabel() {
  return getCurrentEngineEntry()?.label || "Engine";
}

/** The current engine's descriptor, or null. */
function engineEntry() {
  return getCurrentEngineEntry();
}

/** Whether the current engine exposes settings the interface can edit. */
function engineConfigurable() {
  return engineEntry()?.configurable === true;
}

/** Whether the current engine can delete models through the app. */
function engineCanDelete() {
  return engineEntry()?.can_delete_models !== false;
}

// --------------------------------------------------------------------------- #
// Engine chooser
// --------------------------------------------------------------------------- #

/**
 * Paint the bold chooser cards and mark the selected one.
 */
function renderChooser() {
  const engines = getEngines();
  const status = getOllamaStatus();
  const selected = engineId();

  chooserHintEl.textContent =
    "This choice applies to the whole app: the model list, the option editor " +
    "and the run all use the selected engine.";

  if (engines.length === 0) {
    chooserEl.innerHTML = "";
    return;
  }

  chooserEl.innerHTML = engines
    .map((engine) => {
      const isCurrent = engine.id === selected;

      // Only the selected engine's status is fresh in shared state.
      let dotClass = "is-unknown";
      let stateText = "not checked";

      if (isCurrent && status) {
        if (engine.requires_server === false) {
          // A per-run engine has no resident server: report the install, not
          // a running state that would only ever read "not running".
          if (status.installed) {
            dotClass = "is-good";
            stateText = "installed";
          } else {
            dotClass = "is-bad";
            stateText = "not found";
          }
        } else if (status.running) {
          dotClass = "is-good";
          stateText = "running";
        } else if (status.installed) {
          dotClass = "is-bad";
          stateText = "not running";
        } else {
          dotClass = "is-bad";
          stateText = "not found";
        }
      }

      return `
        <button type="button" class="engine-card${isCurrent ? " is-selected" : ""}"
                data-engine-id="${escapeHtml(engine.id)}" role="radio"
                aria-checked="${isCurrent}">
          <span class="engine-card-check" aria-hidden="true">✓</span>
          <span class="engine-card-name">
            ${escapeHtml(engine.label)}
            <span class="engine-card-tag">${escapeHtml(engine.id)}</span>
          </span>
          <span class="engine-card-state">
            <span class="server-dot ${dotClass}" aria-hidden="true"></span>
            ${escapeHtml(stateText)}
          </span>
          <span class="engine-card-desc">${escapeHtml(
            ENGINE_NOTES[engine.id] || "An inference backend."
          )}</span>
        </button>`;
    })
    .join("");

  chooserEl.querySelectorAll("[data-engine-id]").forEach((card) => {
    card.addEventListener("click", () => setCurrentEngine(card.dataset.engineId));
  });
}

// --------------------------------------------------------------------------- #
// Per-engine configuration
// --------------------------------------------------------------------------- #

/** The path separator to show between model folders, by platform. */
function osSep() {
  return navigator.platform.toLowerCase().includes("win") ? ";" : ":";
}

/**
 * Render the configuration block for the current engine.
 *
 * llama.cpp is editable (installation folder, model folders, host); the other
 * engines are shown read-only with a pointer to where their facts come from.
 */
async function renderConfig() {
  if (!engineId()) {
    configBodyEl.innerHTML = "";
    configApplyEl.hidden = true;
    return;
  }

  if (!engineConfigurable()) {
    configApplyEl.hidden = true;
    configMetaEl.textContent = "read from the environment";

    configBodyEl.innerHTML = `
      <p class="engine-config-note">
        <strong>${escapeHtml(engineLabel())}</strong> reads its address from the
        <code>OLLAMA_HOST</code> environment variable (default
        <code>http://127.0.0.1:11434</code>). There is nothing to set here.
      </p>`;
    return;
  }

  let view = null;

  try {
    const { data } = await api("/api/settings");
    view = data?.[engineId()] || null;
  } catch (error) {
    view = null;
  }

  view = view || {};

  configApplyEl.hidden = false;
  configMetaEl.textContent = "saved in this app, wins over the environment";

  configBodyEl.innerHTML = `
    <div class="engine-config-form">
      <label class="field">
        <span class="field-label">llama.cpp installation folder</span>
        <input class="input is-path" id="cfg-home" dir="ltr" type="text"
               value="${escapeHtml(view.home || "")}"
               placeholder="…\\llama-cpp\\…   (the folder holding llama-server.exe)">
        <span class="field-hint">
          The folder that holds <code>llama-server.exe</code>. LLMBench runs the
          server from here and reads its version from the executable.
        </span>
      </label>

      <label class="field">
        <span class="field-label">Model folders (one per line)</span>
        <textarea class="input is-path" id="cfg-model-dirs" dir="ltr"
                  placeholder="D:\\AI-Model\\MODELS"></textarea>
        <span class="field-hint">
          Each folder is scanned, including its subfolders, for <code>.gguf</code>
          files. Every file found becomes a selectable model.
        </span>
      </label>

      <label class="field">
        <span class="field-label">Server address</span>
        <input class="input is-path" id="cfg-host" dir="ltr" type="text"
               value="${escapeHtml(view.host || "")}"
               placeholder="http://127.0.0.1:8082">
        <span class="field-hint">
          Where the spawned server listens. Keep the default unless you run
          several benches at once.
        </span>
      </label>
    </div>`;

  document.getElementById("cfg-model-dirs").value = (view.model_dirs || []).join("\n");

  const envBits = [];

  if (view.env_home) envBits.push(`LLAMA_CPP_HOME=${view.env_home}`);
  if (view.env_model_dirs?.length) envBits.push(`LLAMA_CPP_MODELS=${view.env_model_dirs.join(osSep())}`);
  if (view.env_host) envBits.push(`LLAMA_CPP_HOST=${view.env_host}`);

  if (envBits.length) {
    configBodyEl.insertAdjacentHTML(
      "beforeend",
      `<div class="engine-config-env">environment: ${envBits.map(escapeHtml).join("  ·  ")}</div>`
    );
  }
}

/**
 * Apply the llama.cpp configuration: validate and save it, then re-read the
 * engine's facts so the whole station reflects the new paths.
 */
async function applyConfig() {
  if (!engineConfigurable()) {
    return;
  }

  const home = document.getElementById("cfg-home");
  const dirs = document.getElementById("cfg-model-dirs");
  const host = document.getElementById("cfg-host");

  const payload = {
    home: home.value.trim(),
    model_dirs: dirs.value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean),
    host: host.value.trim(),
  };

  const restore = setButtonBusy(configApplyEl);

  try {
    await postJson(`/api/settings/${encodeURIComponent(engineId())}`, payload);
    toast("success", "llama.cpp settings saved", "Re-reading the server and its models.");
    await refreshServer();
    await renderConfig();
    renderServerCards();
    renderModelTable();
  } catch (error) {
    toast("error", "Could not save the settings", error.message);
  } finally {
    restore();
  }
}

// --------------------------------------------------------------------------- #
// Server
// --------------------------------------------------------------------------- #

/** The raw status dict, or null. */
function statusRaw() {
  return getOllamaStatus();
}

/**
 * Paint the server cards and the state line, adapting the cards to the engine.
 */
function renderServerCards() {
  const status = describeServer();
  const raw = statusRaw();
  const id = engineId();

  const cards =
    id === "llama-cpp"
      ? [
          {
            // llama.cpp has no resident server, so there is nothing to be
            // "running" or "stopped": only whether the install is found.
            label: "Status",
            value: raw?.installed ? "Installed" : "Not found",
            tone: raw?.installed ? "is-good" : "is-bad",
            sub: "a server spawns per model during a run",
          },
          {
            label: "Version",
            value: raw?.version || "—",
            tone: raw?.installed ? "" : "is-unknown",
            sub: "read from the executable",
          },
          {
            label: "Executable",
            value: raw?.executable || "not found",
            tone: raw?.installed ? "" : "is-unknown",
            sub: raw?.home || "set the installation folder",
          },
          {
            label: "Models",
            value: raw?.model_count ?? 0,
            tone: raw?.model_count ? "" : "is-unknown",
            sub: `in ${raw?.model_dirs?.length ?? 0} configured folder(s)`,
          },
        ]
      : [
          {
            label: "Server",
            value: status.tone === "good" ? "running" : status.tone === "bad" ? "stopped" : "unknown",
            tone: status.tone,
            sub: status.note,
          },
          {
            label: "Server version",
            value: raw?.running ? raw.version || "—" : "—",
            tone: raw?.running ? "" : "is-unknown",
            sub: "reported by /api/version",
          },
          {
            label: "Local CLI",
            value:
              raw?.client_version?.client ||
              (raw?.installed ? "installed" : "not found"),
            tone: raw?.installed ? "" : "is-unknown",
            sub: raw?.client_version?.server
              ? `CLI's server: v${raw.client_version.server}`
              : "from the engine's executable",
          },
          {
            label: "Host",
            value: raw?.host || "—",
            tone: "",
            sub: "engine host",
          },
        ];

  cardsEl.innerHTML = cards
    .map(
      (card) => `
        <div class="server-card ${card.tone === "good" ? "is-good" : card.tone === "bad" ? "is-bad" : card.tone === "unknown" ? "is-unknown" : ""}">
          <div class="server-card-label">${escapeHtml(card.label)}</div>
          <div class="server-card-value" dir="ltr">${escapeHtml(card.value)}</div>
          <div class="server-card-sub" dir="auto">${escapeHtml(card.sub)}</div>
        </div>`
    )
    .join("");

  const titleEl = document.getElementById("station-engine-title");

  if (titleEl) {
    // A per-run engine has no server to speak of between runs, so the block
    // is just named after the engine; a resident one is its server.
    titleEl.textContent =
      engineEntry()?.requires_server === false
        ? engineLabel()
        : `${engineLabel()} server`;
  }

  stateLineEl.innerHTML = `
    <span class="state-line-title ${status.tone === "good" ? "is-good" : status.tone === "bad" ? "is-bad" : ""}">
      ${escapeHtml(status.title)}
    </span>
    <span class="state-line-note" dir="auto">${escapeHtml(status.note)}</span>`;

  // llama.cpp has no server to start on its own; it is spawned per model.
  if (startBtn) {
    startBtn.hidden = id === "llama-cpp";
  }
}

// --------------------------------------------------------------------------- #
// Models
// --------------------------------------------------------------------------- #

/**
 * Paint the model table, joining in whether each model is resident in memory.
 */
function renderModelTable() {
  const catalog = getModelCatalog();
  const loaded = getLoadedModels();

  modelCountEl.textContent = catalog.length
    ? `${catalog.length} model${catalog.length > 1 ? "s" : ""}`
    : "none";

  if (catalog.length === 0) {
    modelRowsEl.innerHTML = "";
    modelsEmptyEl.classList.remove("is-hidden");
    modelsEmptyEl.textContent =
      engineId() === "llama-cpp"
        ? "No .gguf files found. Set the model folders above and re-scan."
        : engineCanDelete()
          ? "No models installed. Pull one with 'ollama pull <name>', then refresh."
          : "No models found.";
    return;
  }

  modelsEmptyEl.classList.add("is-hidden");

  modelRowsEl.innerHTML = catalog
    .map((model) => {
      const details = model.details || {};

      // Ollama reports family / parameter count / quantization; llama.cpp
      // reports the folder the file was found under.
      const detailBits = [
        details.family ? String(details.family) : null,
        details.parameter_size ? `${details.parameter_size} params` : null,
        details.quantization_level ? String(details.quantization_level) : null,
        details.relative_folder ? `…\\${details.relative_folder}` : null,
        details.folder && !details.family && !details.relative_folder ? details.folder : null,
      ].filter(Boolean);

      const resident = loaded.includes(model.name);
      const memory = resident ? "in memory" : "—";

      return `
        <tr>
          <td class="model-name-cell" dir="ltr">${escapeHtml(model.name)}</td>
          <td class="is-number">${escapeHtml(humanBytes(model.size))}</td>
          <td class="model-detail-cell" dir="ltr">${escapeHtml(detailBits.join(" · ") || "—")}</td>
          <td class="is-number" style="font-family:var(--font-mono); font-size:11.5px; color:var(--text-dim)">
            ${escapeHtml(humanDateTime(model.modified))}
          </td>
          <td>
            <span class="vram-badge${resident ? " is-loaded" : ""}" dir="ltr">
              ${escapeHtml(memory)}
            </span>
          </td>
          <td>
            <div class="row-actions">
              <button type="button" class="btn btn-sm" data-model-act="bench"
                      data-model-name="${escapeHtml(model.name)}">Bench it</button>
              ${
                resident
                  ? `<button type="button" class="btn btn-sm" data-model-act="unload"
                       data-model-name="${escapeHtml(model.name)}">Unload</button>`
                  : ""
              }
              ${
                engineCanDelete()
                  ? `<button type="button" class="btn btn-sm btn-danger-ghost" data-model-act="remove"
                        data-model-name="${escapeHtml(model.name)}">Delete</button>`
                  : ""
              }
            </div>
          </td>
        </tr>`;
    })
    .join("");

  modelRowsEl.querySelectorAll("[data-model-act]").forEach((button) => {
    button.addEventListener("click", () =>
      onModelAction(button.dataset.modelAct, button.dataset.modelName, button)
    );
  });
}

/**
 * Handle a model row action.
 *
 * @param {"bench"|"unload"|"remove"} act
 * @param {string} name Model name and tag.
 * @param {HTMLButtonElement} button
 */
async function onModelAction(act, name, button) {
  const base = `/api/engines/${encodeURIComponent(engineId())}`;

  if (act === "bench") {
    goToSetup(name);
    return;
  }

  if (act === "remove") {
    if (!window.confirm(`Delete ${name}? This cannot be undone.`)) {
      return;
    }

    const restore = setButtonBusy(button);

    try {
      await postJson(`${base}/models/remove`, { model: name });
      toast("success", "Model deleted", name);
      await refreshServer();
      renderModelTable();
    } catch (error) {
      toast("error", "Could not delete the model", error.message);
    } finally {
      restore();
    }
    return;
  }

  if (act === "unload") {
    const restore = setButtonBusy(button);

    try {
      await postJson(`${base}/models/unload`, { model: name });
      toast("success", "Model unloaded", name);
      await fetchModels();
      renderModelTable();
    } catch (error) {
      toast("error", "Could not unload the model", error.message);
    } finally {
      restore();
    }
  }
}

// --------------------------------------------------------------------------- #
// Wiring
// --------------------------------------------------------------------------- #

/** Repaint the whole station from the shared state. */
function render() {
  renderChooser();
  renderServerCards();
  renderModelTable();
}

/**
 * Load the station's data and paint it.
 */
async function load() {
  await refreshServer();
  render();
  await renderConfig();

  metaEl.textContent = `checked ${new Date().toLocaleTimeString()}`;
}

/**
 * Bind the station's controls.
 */
function bind() {
  document.getElementById("btn-engine-refresh").addEventListener("click", async () => {
    const restore = setButtonBusy(document.getElementById("btn-engine-refresh"));

    try {
      await load();
    } finally {
      restore();
    }
  });

  document.getElementById("btn-engine-models-refresh").addEventListener("click", async () => {
    const restore = setButtonBusy(document.getElementById("btn-engine-models-refresh"));

    try {
      await fetchModels();
      renderModelTable();
    } finally {
      restore();
    }
  });

  configApplyEl.addEventListener("click", applyConfig);

  startBtn.addEventListener("click", async () => {
    const restore = setButtonBusy(startBtn);

    try {
      const { data } = await postJson(
        `/api/engines/${encodeURIComponent(engineId())}/start`,
        {}
      );
      toast(
        "success",
        `${engineLabel()} started`,
        data.version ? `Server version ${data.version}.` : "The API is answering."
      );
      await load();
    } catch (error) {
      toast("error", `Could not start ${engineLabel()}`, error.message);
    } finally {
      restore();
    }
  });

  stopBtn.addEventListener("click", async () => {
    const verb =
      engineId() === "llama-cpp"
        ? "stop the llama.cpp server and free its memory?"
        : `stop the ${engineLabel()} server? Loaded models will be dropped.`;

    if (!window.confirm(verb)) {
      return;
    }

    const restore = setButtonBusy(stopBtn);

    try {
      const res = await postJson(`/api/engines/${encodeURIComponent(engineId())}/stop`, {});

      if (res.restarted) {
        toast(
          "info",
          `${engineLabel()} respawned`,
          "The server process came right back — something supervises it."
        );
      } else {
        toast("success", `${engineLabel()} stopped`);
      }

      await load();
    } catch (error) {
      toast("error", `Could not stop ${engineLabel()}`, error.message);
    } finally {
      restore();
    }
  });
}

/**
 * Bind the station and register its lazy load.
 */
export function initEnginePanel() {
  bind();

  onEnginesChange(async () => {
    await load();
  });

  onTabFirstOpen("engine", load);
}
