// Shared engine server reader.
//
// Both the command-bar chip and the stations that show server facts go through
// here, so a status or model refresh happens once and every interested panel
// re-renders from the same shared state — always for the engine currently
// selected in lib/engines.js. Listeners subscribe with onServerChange and are
// told which pieces moved, so each panel redraws only its own corner of the
// page.

import { api } from "./api.js";
import { getCurrentEngine, getCurrentEngineEntry, onEnginesChange } from "./engines.js";
import {
  getOllamaStatus,
  getLoadedModels,
  getModelCatalog,
  setLoadedModels,
  setModelCatalog,
  setOllamaStatus,
} from "./state.js";

const listeners = new Set();

/**
 * Subscribe to server state changes.
 *
 * @param {(moved: {status: boolean, models: boolean}) => void} listener
 * @returns {Function} Call to unsubscribe.
 */
export function onServerChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function notify(moved) {
  listeners.forEach((listener) => listener(moved));
}

/** URL prefix of the current engine's endpoints, or null before selection. */
function engineBase() {
  const id = getCurrentEngine();
  return id ? `/api/engines/${encodeURIComponent(id)}` : null;
}

/** The current engine's display label. */
function engineLabel() {
  return getCurrentEngineEntry()?.label || "engine";
}

/** Paint the command-bar chip and dot from the shared status. */
function paintChip() {
  const dot = document.getElementById("server-dot");
  const label = document.getElementById("server-label");
  const status = getOllamaStatus();
  const name = engineLabel().toLowerCase();

  if (!dot || !label) {
    return;
  }

  if (!status) {
    dot.className = "server-dot is-unknown";
    label.textContent = "server unknown";
    return;
  }

  // An engine that spawns its own server per run has no resident server, so
  // between runs it is never "not running": the chip reports whether it is
  // installed instead.
  if (getCurrentEngineEntry()?.requires_server === false) {
    if (status.installed) {
      dot.className = "server-dot is-good";
      label.textContent = status.version ? `${name} v${status.version}` : `${name} installed`;
    } else {
      dot.className = "server-dot is-bad";
      label.textContent = `${name} not found`;
    }

    return;
  }

  if (status.running) {
    dot.className = "server-dot is-good";
    label.textContent = status.version ? `${name} v${status.version}` : `${name} running`;
  } else if (status.installed) {
    dot.className = "server-dot is-bad";
    label.textContent = `${name} not running`;
  } else {
    dot.className = "server-dot is-bad";
    label.textContent = `${name} not found`;
  }
}

/**
 * Fetch the current engine's server status into shared state and repaint the
 * chip.
 *
 * @returns {object|null} The status dict, or null on failure.
 */
export async function fetchServerStatus() {
  const base = engineBase();

  if (!base) {
    setOllamaStatus(null);
  } else {
    try {
      const { data } = await api(`${base}/status`);
      setOllamaStatus(data);
    } catch (error) {
      setOllamaStatus(null);
    }
  }

  paintChip();
  notify({ status: true, models: false });

  return getOllamaStatus();
}

/**
 * Fetch the installed model catalog and the resident list into shared state,
 * for the current engine.
 *
 * A failing fetch clears the catalog: an empty list the model tick-list can
 * explain with the status, rather than a stale one that names models that are
 * gone.
 *
 * @returns {{catalog: Array<object>, loaded: Array<object>}} What came back.
 */
export async function fetchModels() {
  const base = engineBase();
  let catalog = [];
  let loaded = [];

  if (base) {
    try {
      const { data } = await api(`${base}/models`);
      catalog = data || [];
    } catch (error) {
      catalog = [];
    }

    try {
      const { data } = await api(`${base}/models/running`);
      loaded = data || [];
    } catch (error) {
      loaded = [];
    }
  }

  setModelCatalog(catalog);
  setLoadedModels(loaded.map((model) => model.name));
  notify({ status: false, models: true });

  return { catalog, loaded };
}

/**
 * Refresh status and models together and update the tab badges.
 *
 * @returns {Promise<void>}
 */
export async function refreshServer() {
  await Promise.all([fetchServerStatus(), fetchModels()]);
  paintBadges();
}

/** Update the tab badges from the shared state. */
function paintBadges() {
  const engineBadge = document.getElementById("badge-engine");

  if (engineBadge) {
    engineBadge.textContent = getModelCatalog().length ? String(getModelCatalog().length) : "";
  }
}

/**
 * Paint the badges once the state first exists. Called from the bootstrap.
 */
export function paintServerBadges() {
  paintBadges();
}

/**
 * Return a one-line human explanation of the engine server for a panel to
 * show.
 *
 * @returns {{tone: "good"|"bad"|"unknown", title: string, note: string}}
 */
export function describeServer() {
  const status = getOllamaStatus();
  const name = engineLabel();

  if (!status) {
    return {
      tone: "unknown",
      title: "Unknown",
      note: `The status endpoint did not answer; the app cannot see ${name}.`,
    };
  }

  // An engine that spawns its own server per run has no resident server to be
  // "running", so only whether the installation is present is worth stating.
  if (getCurrentEngineEntry()?.requires_server === false) {
    return status.installed
      ? {
          tone: "good",
          title: "Installed",
          note: `${name} is installed. A server is spawned per model during a run and stopped when it finishes.`,
        }
      : {
          tone: "bad",
          title: "Not installed",
          note: `No ${name} installation was found. Set its installation folder in the Configuration block.`,
        };
  }

  if (status.running) {
    return {
      tone: "good",
      title: "Running",
      note: `Server answers at ${status.host}${
        status.version ? ` · v${status.version}` : ""
      }.`,
    };
  }

  if (status.installed) {
    return {
      tone: "bad",
      title: "Installed, not running",
      note: `${name} is installed on this machine, but its server is not answering.`,
    };
  }

  return {
    tone: "bad",
    title: "Not found",
    note:
      `No ${name} server answers at the configured host and no local ` +
      `installation was found. Install it or set its environment variables ` +
      `(OLLAMA_HOST, LLAMA_CPP_HOME).`,
  };
}

/**
 * Whether a model name matches one that is resident in memory.
 *
 * Ollama reports loaded models with an explicit tag, so a bare name such as
 * "llama3" also matches the ":latest" form it was loaded under. Engines that
 * report exact file names match exactly.
 *
 * @param {string} name Model name or tag.
 * @returns {boolean}
 */
export function isModelLoaded(name) {
  return getLoadedModels().some(
    (loaded) =>
      loaded === name || (!name.includes(":") && loaded === `${name}:latest`)
  );
}

/**
 * Repaint when the engine selection moves.
 *
 * The chip is repainted here; the passed refresh is awaited so the shared
 * state is replaced with the new engine's facts before panels redraw.
 *
 * @param {() => Promise<void>} refresh The bootstrap's refreshServer.
 */
export function bindEngineSwitching(refresh) {
  onEnginesChange(async () => {
    paintChip();
    await refresh();
  });
}
