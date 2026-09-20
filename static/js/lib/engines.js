// Engine registry for the interface.
//
// The app can benchmark against several inference engines; this module owns
// the list and which one the dashboard is currently pointed at. Everything
// engine-scoped — the server chip, the model lists, the option editor, the
// run payload — reads the current engine from here, and re-renders when the
// selection moves via onEnginesChange.

import { api } from "./api.js";

const STORAGE_KEY = "llmbench.engine";

const listeners = new Set();

/** @type {Array<{id: string, label: string, running: boolean, can_delete_models: boolean}>} */
let engines = [];

/** @type {string|null} */
let currentId = null;

/**
 * Subscribe to engine list or selection changes.
 *
 * @param {() => void} listener
 * @returns {Function} Call to unsubscribe.
 */
export function onEnginesChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function notify() {
  listeners.forEach((listener) => listener());
}

/**
 * Return the known engines, in display order.
 *
 * @returns {Array<{id: string, label: string, running: boolean, can_delete_models: boolean}>}
 */
export function getEngines() {
  return engines;
}

/**
 * Return the current engine's id, or null before the first fetch.
 *
 * @returns {string|null}
 */
export function getCurrentEngine() {
  return currentId;
}

/**
 * Return the current engine's descriptor, or null before the first fetch.
 *
 * @returns {object|null}
 */
export function getCurrentEngineEntry() {
  return engines.find((engine) => engine.id === currentId) || null;
}

/**
 * Select a different engine and tell every listener.
 *
 * The choice persists in localStorage, so the next visit lands on the same
 * engine.
 *
 * @param {string} id Engine id from the registry.
 */
export function setCurrentEngine(id) {
  if (id === currentId || !engines.some((engine) => engine.id === id)) {
    return;
  }

  currentId = id;
  localStorage.setItem(STORAGE_KEY, id);
  notify();
}

/**
 * Fetch the engine registry into shared state.
 *
 * The stored selection wins when it still exists; otherwise the first
 * engine is selected. Returns the resolved id.
 *
 * @returns {Promise<string|null>} The selected engine id.
 */
export async function fetchEngines() {
  try {
    const { data } = await api("/api/engines");
    engines = data || [];
  } catch (error) {
    engines = [];
    notify();
    return null;
  }

  const stored = localStorage.getItem(STORAGE_KEY);
  const wanted =
    stored && engines.some((engine) => engine.id === stored)
      ? stored
      : engines[0]?.id || null;

  const changed = wanted !== currentId;
  currentId = wanted;

  if (changed) {
    notify();
  }

  return currentId;
}
