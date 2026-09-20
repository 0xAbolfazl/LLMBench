// The manual configuration editor, opened as a modal.
//
// The option fields are built from the schema the server publishes, so the
// browser can never drift from what the server accepts. Every option has three
// states, not two — set to a value, or not set at all — so a blank field means
// "leave the model's default alone" and is never sent. Booleans therefore
// render as a three-way select rather than a checkbox.

import { escapeHtml } from "../lib/format.js";
import { openModal } from "../lib/modal.js";

// Sampling is what most comparisons vary, so it is the one group open on load.
const OPEN_BY_DEFAULT = "Sampling";

let schema = [];
let modal = null;
let editingExtras = {};

/**
 * Build one option's input.
 *
 * @param {object} option Schema entry.
 * @returns {string} HTML markup.
 */
function optionField(option) {
  const key = escapeHtml(option.key);
  const hint = option.hint ? `<span class="field-hint">${escapeHtml(option.hint)}</span>` : "";

  let control;

  if (option.type === "bool") {
    control = `
      <select class="input" data-option="${key}" dir="ltr">
        <option value="">not set</option>
        <option value="true">true</option>
        <option value="false">false</option>
      </select>`;
  } else if (option.type === "list") {
    control = `<input class="input" dir="ltr" data-option="${key}" autocomplete="off"
      spellcheck="false" placeholder="${escapeHtml(option.placeholder || "")}">`;
  } else {
    const bounds = [
      option.min !== undefined ? `min="${escapeHtml(option.min)}"` : "",
      option.max !== undefined ? `max="${escapeHtml(option.max)}"` : "",
      option.step !== undefined ? `step="${escapeHtml(option.step)}"` : "",
    ]
      .filter(Boolean)
      .join(" ");

    control = `<input class="input" type="number" dir="ltr" data-option="${key}" ${bounds}
      placeholder="${escapeHtml(option.placeholder || "")}">`;
  }

  return `
    <label class="opt-field" data-option-field="${key}">
      <span class="field-label" dir="ltr">${escapeHtml(option.label || option.key)}</span>
      ${control}
      ${hint}
    </label>`;
}

/**
 * Build the body markup of the editor modal from the schema.
 *
 * @returns {string} HTML markup.
 */
function editorBody() {
  const groups = [];

  schema.forEach((option) => {
    const title = option.group || "Options";
    let group = groups.find((entry) => entry.title === title);

    if (!group) {
      group = { title, options: [] };
      groups.push(group);
    }

    group.options.push(option);
  });

  const groupsHtml = groups
    .map((group) => {
      const open = group.title === OPEN_BY_DEFAULT ? " is-open" : "";
      return `
        <section class="opt-group${open}" data-group="${escapeHtml(group.title)}">
          <button type="button" class="opt-group-head" aria-expanded="${open ? "true" : "false"}">
            <span class="opt-caret" aria-hidden="true">▶</span>
            ${escapeHtml(group.title)}
            <span class="opt-group-badge" data-badge>0 set</span>
          </button>
          <div class="opt-group-body">
            ${group.options.map(optionField).join("")}
          </div>
        </section>`;
    })
    .join("");

  return `
    <label class="field">
      <span class="field-label">Name</span>
      <input class="input" id="editor-name" dir="ltr" placeholder="baseline"
             autocomplete="off" spellcheck="false">
      <span class="field-hint">Shown in the results table.</span>
    </label>
    <div class="opt-groups" id="editor-groups">${groupsHtml}</div>`;
}

/** Mark the fields that hold a value and refresh the per-group counters. */
function markFilled() {
  const groupsEl = document.getElementById("editor-groups");
  const countEl = document.getElementById("editor-count");

  if (!groupsEl) {
    return;
  }

  let total = 0;

  groupsEl.querySelectorAll(".opt-group").forEach((group) => {
    let set = 0;

    group.querySelectorAll("[data-option]").forEach((input) => {
      const filled = input.value.trim() !== "";
      input.closest(".opt-field").classList.toggle("is-set", filled);

      if (filled) {
        set += 1;
      }
    });

    const badge = group.querySelector("[data-badge]");
    badge.textContent = set === 0 ? "0 set" : `${set} set`;
    badge.classList.toggle("is-set", set > 0);
    total += set;
  });

  if (countEl) {
    countEl.textContent = total === 0 ? "no options set" : `${total} option${total > 1 ? "s" : ""} set`;
  }
}

/** Empty every field in the open editor. */
function resetFields() {
  const nameEl = document.getElementById("editor-name");
  const groupsEl = document.getElementById("editor-groups");

  if (nameEl) {
    nameEl.value = "";
  }

  groupsEl.querySelectorAll("[data-option]").forEach((input) => {
    input.value = "";
  });

  markFilled();
}

/** Fill the open editor from a configuration, keeping any unlisted options. */
function fillFields(config) {
  resetFields();

  const nameEl = document.getElementById("editor-name");
  const groupsEl = document.getElementById("editor-groups");
  nameEl.value = config.name || "";

  editingExtras = {};

  Object.entries(config.options || {}).forEach(([key, value]) => {
    const input = groupsEl.querySelector(`[data-option="${CSS.escape(key)}"]`);

    if (!input) {
      editingExtras[key] = value;
      return;
    }

    input.value = Array.isArray(value) ? value.join(", ") : String(value);

    // Reveal the group holding a set option, so editing does not start with
    // the values hidden behind a collapsed header.
    const group = input.closest(".opt-group");
    group.classList.add("is-open");
    group.querySelector(".opt-group-head").setAttribute("aria-expanded", "true");
  });

  markFilled();
}

/**
 * Read the open editor into a configuration.
 *
 * Values are left as the strings the fields hold; the server coerces them
 * using the same schema it published, so the browser never has to guess a
 * type. Options with no field (newer Ollama keys) are merged back from the
 * extras kept when the configuration was filled.
 *
 * @returns {{name: string, options: object}} The configuration.
 */
export function readEditor() {
  const nameEl = document.getElementById("editor-name");
  const options = { ...editingExtras };

  schema.forEach((option) => {
    const input = document.querySelector(`[data-option="${CSS.escape(option.key)}"]`);
    const value = input ? input.value.trim() : "";

    if (value === "") {
      return;
    }

    options[option.key] = value;
  });

  return { name: nameEl.value.trim(), options };
}

/**
 * Open the editor modal in "add" or "edit" mode.
 *
 * @param {"add"|"edit"} mode Which action the footer button performs.
 * @param {object|null} config Configuration to edit, in edit mode.
 * @param {{onSave: Function, onCancel: Function}} callbacks Run when the
 *     caller's Save or Reset resolves the edit; Save receives the read draft.
 */
export function openEditor(mode, config, { onSave, onCancel }) {
  const title = mode === "add" ? "Add a configuration" : `Edit “${config.name}”`;
  const saveLabel = mode === "add" ? "Add" : "Save";

  modal = openModal(
    title,
    editorBody(),
    `
      <span class="modal-foot-note" id="editor-count"></span>
      <div class="btn-row">
        <button type="button" class="btn" id="editor-reset">Reset</button>
        <button type="button" class="btn btn-accent" id="editor-save">${saveLabel}</button>
      </div>`
  );

  if (mode === "edit") {
    fillFields(config);
  } else {
    editingExtras = {};
  }

  // Group headers fold and unfold their option grids.
  modal.el.querySelectorAll(".opt-group-head").forEach((head) => {
    head.addEventListener("click", () => {
      const group = head.closest(".opt-group");
      const open = group.classList.toggle("is-open");
      head.setAttribute("aria-expanded", open ? "true" : "false");
    });
  });

  // A field that gets a value is marked as it is typed, and the counters
  // follow it, so a collapsed group still reports what it holds.
  modal.el.addEventListener("input", markFilled);
  modal.el.addEventListener("change", markFilled);
  markFilled();

  modal.el.querySelector("#editor-reset").addEventListener("click", () => {
    resetFields();
    editingExtras = {};
  });

  modal.el.querySelector("#editor-save").addEventListener("click", () => {
    const draft = readEditor();
    modal.close();
    onSave(draft);
  });

  modal.onClose(() => {
    modal = null;
    onCancel?.();
  });

  const nameEl = modal.el.querySelector("#editor-name");
  window.setTimeout(() => nameEl.focus(), 30);
}

/**
 * Store the option schema the editor is built from. Called once at bootstrap
 * after the schema endpoint answers.
 *
 * @param {Array<object>} options Schema entries.
 */
export function setSchema(options) {
  schema = options;
}
