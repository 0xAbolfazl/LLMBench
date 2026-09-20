// The set of configurations the workbench is holding, plus the colour each
// one is drawn in.
//
// Configurations are identified by a generated id rather than by their name:
// the name is editable and can collide, while cards, table rows and bars all
// need to point at the same entry across re-renders.

// Series colours, cycled per configuration. A configuration keeps its colour in
// the card list, the summary table and every bar, so the same run is
// recognisable wherever it appears.
const SERIES_COLORS = [
  "#5AA9F5",
  "#56D89A",
  "#A78BFA",
  "#F0B457",
  "#F2685F",
  "#4FD1C5",
  "#F472B6",
  "#93C5FD",
  "#FBBF24",
  "#34D399",
  "#C084FC",
  "#FB7185",
];

let configurations = [];
let nextId = 1;

/**
 * Return the drawing colour for a configuration at a given position.
 *
 * @param {number} index Zero-based position in the list.
 * @returns {string} CSS colour.
 */
export function seriesColor(index) {
  return SERIES_COLORS[index % SERIES_COLORS.length];
}

/**
 * Return the current configurations.
 *
 * @returns {Array<{id: number, name: string, options: object}>} Configurations.
 */
export function getConfigurations() {
  return configurations;
}

/**
 * Return one configuration by id.
 *
 * @param {number} id Configuration id.
 * @returns {{id: number, name: string, options: object}|undefined} Match.
 */
export function getConfiguration(id) {
  return configurations.find((config) => config.id === Number(id));
}

/**
 * Find the position of a configuration, so it can be given its colour.
 *
 * @param {string} name Configuration name as it appears in a result.
 * @returns {number} Zero-based index, or -1 when it is no longer in the list.
 */
export function indexOfName(name) {
  return configurations.findIndex((config) => config.name === name);
}

/**
 * Give a configuration a name no other one is using.
 *
 * Names are what the results table labels its rows with, so two identical names
 * would make two rows impossible to tell apart.
 *
 * @param {string} name Requested name, possibly blank.
 * @param {number|null} [ignoreId] Configuration allowed to keep its own name.
 * @returns {string} A unique name.
 */
function uniqueName(name, ignoreId = null) {
  const base = (name || "").trim() || `config_${configurations.length + 1}`;
  const taken = new Set(
    configurations.filter((config) => config.id !== ignoreId).map((config) => config.name)
  );

  if (!taken.has(base)) {
    return base;
  }

  let suffix = 2;
  while (taken.has(`${base} (${suffix})`)) {
    suffix += 1;
  }

  return `${base} (${suffix})`;
}

/**
 * Add a configuration.
 *
 * @param {{name?: string, options: object}} config Name and Ollama options.
 * @returns {object} The stored configuration, with its id and final name.
 */
export function addConfiguration({ name, options }) {
  const stored = {
    id: nextId++,
    name: uniqueName(name),
    options: { ...options },
  };

  configurations.push(stored);
  return stored;
}

/**
 * Replace a configuration's name and options.
 *
 * @param {number} id Configuration id.
 * @param {{name?: string, options: object}} config Replacement values.
 * @returns {object|null} The updated configuration, or null when the id is gone.
 */
export function updateConfiguration(id, { name, options }) {
  const stored = getConfiguration(id);

  if (!stored) {
    return null;
  }

  stored.name = uniqueName(name, stored.id);
  stored.options = { ...options };

  return stored;
}

/**
 * Copy a configuration, placing the copy directly after the original.
 *
 * Duplicating is how a variant is built: keep everything, change one option.
 *
 * @param {number} id Configuration id to copy.
 * @returns {object|null} The copy, or null when the id is gone.
 */
export function duplicateConfiguration(id) {
  const source = getConfiguration(id);

  if (!source) {
    return null;
  }

  const copy = {
    id: nextId++,
    name: uniqueName(source.name),
    options: { ...source.options },
  };

  configurations.splice(configurations.indexOf(source) + 1, 0, copy);

  return copy;
}

/**
 * Remove a configuration.
 *
 * @param {number} id Configuration id.
 */
export function removeConfiguration(id) {
  configurations = configurations.filter((config) => config.id !== Number(id));
}

/** Remove every configuration. */
export function clearConfigurations() {
  configurations = [];
}

/**
 * Append several configurations at once, as an uploaded file provides them.
 *
 * @param {Array<{name?: string, options: object}>} entries Configurations.
 * @returns {number} How many were added.
 */
export function addConfigurations(entries) {
  entries.forEach((entry) => addConfiguration(entry));
  return entries.length;
}

/**
 * Return the configurations in the shape the run endpoint expects.
 *
 * @returns {Array<{name: string, options: object}>} Payload configurations.
 */
export function toPayload() {
  return configurations.map(({ name, options }) => ({ name, options }));
}
