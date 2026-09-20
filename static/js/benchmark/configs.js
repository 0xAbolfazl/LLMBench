// Renderers for the workbench's configuration cards and option chips.

import { escapeHtml } from "../lib/format.js";
import { getConfigurations, seriesColor } from "./store.js";

/**
 * Render one option's value as it appears on a configuration card.
 *
 * Option names are Ollama's own identifiers and values are numbers, so the
 * chip reads left to right regardless of the surrounding text.
 *
 * @param {string} key Option name.
 * @param {*} value Option value.
 * @returns {string} HTML markup.
 */
function optionChip(key, value) {
  const text = Array.isArray(value) ? value.join(", ") : String(value);
  return `<span class="config-chip" dir="ltr" title="${escapeHtml(`${key} = ${text}`)}">
    <b>${escapeHtml(key)}</b> ${escapeHtml(text)}</span>`;
}

/**
 * Render every option of one configuration as chips.
 *
 * @param {object} options Ollama options.
 * @returns {string} HTML markup.
 */
export function optionChips(options) {
  const entries = Object.entries(options || {});

  if (entries.length === 0) {
    return `<span class="config-chip">model defaults</span>`;
  }

  return entries.map(([key, value]) => optionChip(key, value)).join("");
}

/**
 * Paint the configuration cards.
 *
 * @param {HTMLElement} el Container element.
 */
export function renderConfigList(el) {
  const configurations = getConfigurations();

  if (configurations.length === 0) {
    el.innerHTML = `<div class="empty-state">
      No configurations yet. Add one, or leave this empty for the model's own
      defaults.
    </div>`;
    return;
  }

  el.innerHTML = configurations
    .map((config, index) => {
      const color = seriesColor(index);
      return `
        <article class="config-card" style="--series-color:${color}">
          <span class="config-index" dir="ltr">${String(index + 1).padStart(2, "0")}</span>
          <span class="config-name" dir="ltr" title="${escapeHtml(config.name)}">${escapeHtml(config.name)}</span>
          <div class="config-actions">
            <button type="button" class="btn btn-sm" data-config-act="edit" data-config-id="${config.id}">Edit</button>
            <button type="button" class="btn btn-sm" data-config-act="copy" data-config-id="${config.id}">Duplicate</button>
            <button type="button" class="btn btn-sm btn-danger-ghost" data-config-act="remove" data-config-id="${config.id}">Remove</button>
          </div>
          <div class="config-chips">${optionChips(config.options)}</div>
        </article>`;
    })
    .join("");
}
