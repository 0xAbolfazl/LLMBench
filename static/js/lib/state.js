// Shared state between the page's parts.
//
// The setup station and the Ollama station both need the server status (to tell
// "no models installed" apart from "the server is not running") and the model
// catalog. Keeping them here lets the command-bar chip, the model tick-list
// and the model table all read the same facts without importing each other.

/** @type {{installed: boolean, running: boolean, version: string|null, client_version: object|null, host: string}|null} */
let ollamaStatus = null;

/**
 * The installed model catalog, one entry per model:
 * {name, size, modified, details}.
 *
 * @type {Array<{name: string, size: number|null, modified: number|null, details: object}>}
 */
let modelCatalog = [];

/** Names resident in memory right now, from /api/ps. @type {string[]} */
let loadedModels = [];

/**
 * Return the last known runtime status, or null if it was never read.
 *
 * @returns {object|null} Status dict.
 */
export function getOllamaStatus() {
  return ollamaStatus;
}

/**
 * Record the runtime status.
 *
 * @param {object|null} status Status dict, or null when unknown.
 */
export function setOllamaStatus(status) {
  ollamaStatus = status;
}

/**
 * Return the installed model catalog.
 *
 * @returns {Array<object>} Model entries.
 */
export function getModelCatalog() {
  return modelCatalog;
}

/**
 * Record the installed model catalog.
 *
 * @param {Array<object>} catalog Model entries.
 */
export function setModelCatalog(catalog) {
  modelCatalog = catalog || [];
}

/**
 * Return the installed model names, in catalog order.
 *
 * @returns {string[]} Model names.
 */
export function getInstalledModels() {
  return modelCatalog.map((model) => model.name);
}

/**
 * Return the names resident in memory.
 *
 * @returns {string[]} Loaded model names.
 */
export function getLoadedModels() {
  return loadedModels;
}

/**
 * Record the names resident in memory.
 *
 * @param {string[]} names Loaded model names.
 */
export function setLoadedModels(names) {
  loadedModels = names || [];
}
