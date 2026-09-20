// History station: finished runs are saved into the history automatically, and
// this station lists them. The list loads the first time the station is opened
// and again through its refresh button; each run's detail opens inline inside
// its row, and nothing here touches the live run the results area is showing.
//
// The server answers:
//   GET    /api/history          the index, newest first
//   GET    /api/history/<id>     one run in full
//   DELETE /api/history/<id>     remove one run

import { api } from "../lib/api.js";
import { escapeHtml } from "../lib/format.js";
import { setButtonBusy, toast } from "../lib/toast.js";
import { onTabFirstOpen } from "../tabs.js";

const el = document.getElementById("history-body");
const countEl = document.getElementById("history-count");
const refreshBtn = document.getElementById("btn-history-refresh");

// Records already fetched. Refetched through the refresh button, so a run that
// finished while the page sat open shows up then.
let records = [];

/** Whether the noise verdict is known for a record. */
function noiseVerdict(record) {
  // List records carry the verdict at the top level; a full record cached
  // back into the list carries it under summary instead.
  const significant = record.significant ?? record.summary?.significant;

  if (significant === true || significant === false) {
    return escapeHtml(significant ? "difference real" : "within noise");
  }

  return `<span class="is-dim">not measured</span>`;
}

/** Sort the per-entry averages fastest first. */
function rankedAverages(record) {
  const averages = record.summary?.average_output_tokens_per_second || {};

  return Object.entries(averages)
    .map(([name, rate]) => ({ name, rate }))
    .sort((a, b) => b.rate - a.rate);
}

/** Build one row's inline detail: the per-entry averages and verdict. */
function detailMarkup(record) {
  const rows = rankedAverages(record)
    .map(
      (entry) => `
        <div class="history-avg">
          <span class="history-avg-name" dir="ltr">${escapeHtml(entry.name)}</span>
          <span class="history-avg-value" dir="ltr">${Number(entry.rate).toFixed(2)} tok/s${
            entry.name === record.summary.winner ? ' <span class="is-winner">★</span>' : ""
          }</span>
        </div>`
    )
    .join("");

  // The full record carries the significance message under either half of the
  // matrix, so read the one that is present.
  const sig = record.result?.significance || {};
  const message =
    sig.across_models?.message ||
    Object.values(sig.by_model || {}).find((a) => a && a.message)?.message ||
    "";

  return `
    <div class="history-detail-inner">
      ${rows || `<div class="history-avg">No measurable averages.</div>`}
      ${message ? `<div class="history-message" dir="auto">${escapeHtml(message)}</div>` : ""}
      <div>
        <button type="button" class="btn btn-sm btn-danger" data-hact="del" data-hid="${escapeHtml(
          record.id
        )}">Delete</button>
      </div>
    </div>`;
}

/** Render the table from the current records list. */
function render() {
  countEl.textContent = records.length
    ? `${records.length} saved run${records.length > 1 ? "s" : ""}`
    : "empty";

  if (records.length === 0) {
    el.innerHTML = `<div class="empty-state">No saved runs yet. Finished runs land here automatically.</div>`;
    return;
  }

  const rows = records
    .map(
      (record) => `
        <tbody data-hid="${escapeHtml(record.id)}">
          <tr>
            <td class="log-time" dir="ltr">${escapeHtml(record.saved_at || "—")}</td>
            <td class="is-ltr" dir="ltr" title="${escapeHtml(
              [
                record.engine || null,
                (record.models || [record.model].filter(Boolean)).join(", ") || null,
              ]
                .filter(Boolean)
                .join(" · ")
            )}">${escapeHtml(
              [
                record.engine || null,
                (record.models || [record.model].filter(Boolean)).join(", ") || null,
              ]
                .filter(Boolean)
                .join(" · ")
            )}</td>
            <td class="is-number">${
              record.configuration_count ?? record.configurations?.length ?? "—"
            }</td>
            <td class="is-number">${record.prompt_count ?? record.prompts?.length ?? "—"}</td>
            <td class="log-time" dir="ltr" title="${escapeHtml(record.winner ?? "")}">${escapeHtml(
              (record.winner ?? record.summary?.winner) || "—"
            )}</td>
            <td>${noiseVerdict(record)}</td>
            <td>
              <div class="row-actions">
                <button type="button" class="btn btn-sm" data-hact="toggle" data-hid="${escapeHtml(
                  record.id
                )}">Details</button>
              </div>
            </td>
          </tr>
          <tr class="history-detail-row is-hidden">
            <!-- Filled on first open with the full record; a list record has
                 no summary to pre-render from. -->
            <td colspan="7">${record.summary ? detailMarkup(record) : ""}</td>
          </tr>
        </tbody>`
    )
    .join("");

  el.innerHTML = `
    <div class="table-wrap">
      <table class="data-table" aria-label="Saved benchmark runs">
        <thead>
          <tr>
            <th>Saved</th>
            <th>Model(s)</th>
            <th class="is-number">Entries</th>
            <th class="is-number">Prompts</th>
            <th>Fastest</th>
            <th>Noise</th>
            <th></th>
          </tr>
        </thead>
        ${rows}
      </table>
    </div>`;
}

/** Fetch the index and repaint. */
async function load() {
  try {
    records = (await api("/api/history")).data;
    render();
  } catch (error) {
    countEl.textContent = "unavailable";
    el.innerHTML = `<div class="alert">${escapeHtml(
      `Could not load history: ${error.message}`
    )}</div>`;
  }
}

/**
 * Toggle one row's inline detail, loading the full record on first open.
 *
 * @param {HTMLElement} button The button that was pressed.
 */
async function toggleDetail(button) {
  const group = button.closest("tbody");
  const detailRow = group.querySelector(".history-detail-row");

  if (!detailRow.classList.contains("is-hidden")) {
    detailRow.classList.add("is-hidden");
    return;
  }

  detailRow.classList.remove("is-hidden");

  // The list carries only the summary; the full record (with the significance
  // message) arrives on first open and is cached back into the list.
  const id = group.dataset.hid;
  const known = records.find((record) => record.id === id);
  const record = known?.result ? known : (await api(`/api/history/${id}`)).data;
  const position = records.findIndex((entry) => entry.id === id);

  if (position !== -1) {
    records[position] = record;
  }

  detailRow.innerHTML = `<td colspan="7">${detailMarkup(record)}</td>`;
}

/**
 * Remove one saved run, with the same confirm-then-toast pattern the rest of
 * the destructive actions use.
 *
 * @param {HTMLElement} button The button that was pressed.
 */
async function removeRun(button) {
  const group = button.closest("tbody");
  const id = group.dataset.hid;

  if (!window.confirm("Delete this saved run? This cannot be undone.")) {
    return;
  }

  try {
    const { data } = await api(`/api/history/${id}`, { method: "DELETE" });

    if (data?.deleted) {
      toast("success", "Run deleted", id);
    }

    records = records.filter((record) => record.id !== id);
    render();
  } catch (error) {
    toast("error", "Could not delete the run", error.message);
  }
}

function bind() {
  refreshBtn.addEventListener("click", async () => {
    const restore = setButtonBusy(refreshBtn);

    try {
      await load();
    } finally {
      restore();
    }
  });

  el.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-hact]");

    if (!button) {
      return;
    }

    if (button.dataset.hact === "toggle") {
      await toggleDetail(button);
    } else if (button.dataset.hact === "del") {
      await removeRun(button);
    }
  });
}

/** Bind the station and load its list the first time it is opened. */
export function initHistoryPanel() {
  bind();
  onTabFirstOpen("history", load);
}
