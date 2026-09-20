// Log station: the backend writes a line for every benchmark step as it
// happens, and this station reads that log back out. The entries load the
// first time the station is opened and again through its refresh button; the
// level filter re-asks the server, so the parsing and filtering stay on the
// backend where the format is defined.

import { api } from "./lib/api.js";
import { escapeHtml } from "./lib/format.js";
import { setButtonBusy, toast } from "./lib/toast.js";
import { onTabFirstOpen } from "./tabs.js";

const el = document.getElementById("log-body");
const countEl = document.getElementById("log-count");
const metaEl = document.getElementById("log-meta");
const levelEl = document.getElementById("log-level");
const refreshBtn = document.getElementById("btn-log-refresh");
const clearBtn = document.getElementById("btn-log-clear");

let entries = [];

/** Tone class for one severity, so the table's level column scans fast. */
function levelTone(level) {
  if (level === "ERROR") return "log-level-error";
  if (level === "WARNING") return "log-level-warning";
  return "log-level-info";
}

/** Render the table from the current entries. */
function render() {
  countEl.textContent = entries.length ? `${entries.length} entries` : "empty";
  clearBtn.disabled = entries.length === 0;

  if (entries.length === 0) {
    el.innerHTML = `<div class="empty-state">No log entries yet. Run a benchmark to see its steps appear here.</div>`;
    return;
  }

  const rows = entries
    .map((entry) => {
      // The structured details ride along as a tooltip on the message.
      const details = Object.keys(entry.details || {}).length
        ? ` title="${escapeHtml(JSON.stringify(entry.details))}"`
        : "";

      return `
        <tr>
          <td class="log-time" dir="ltr">${escapeHtml(entry.timestamp)}</td>
          <td><span class="log-level ${levelTone(entry.level)}">${escapeHtml(entry.level)}</span></td>
          <td class="log-component" dir="ltr">${escapeHtml(entry.component)}</td>
          <td class="log-action" dir="ltr">${escapeHtml(entry.action)}</td>
          <td class="log-message"${details}>${escapeHtml(entry.message)}</td>
        </tr>`;
    })
    .join("");

  el.innerHTML = `
    <div class="table-wrap">
      <table class="data-table" aria-label="Execution log">
        <thead>
          <tr>
            <th>Time</th>
            <th>Level</th>
            <th>Component</th>
            <th>Action</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

/** Fetch the entries under the current level filter and repaint. */
async function load() {
  const params = new URLSearchParams();

  if (levelEl.value) {
    params.set("level", levelEl.value);
  }

  try {
    const { data } = await api(`/api/logs?${params.toString()}`);
    entries = data.entries;

    metaEl.textContent = data.file.exists
      ? `${data.file.path} · ${data.file.size_bytes} bytes`
      : data.file.path;

    render();
  } catch (error) {
    toast("error", "Could not load the log", error.message);
  }
}

function bind() {
  levelEl.addEventListener("change", load);

  refreshBtn.addEventListener("click", async () => {
    const restore = setButtonBusy(refreshBtn);

    try {
      await load();
    } finally {
      restore();
    }
  });

  clearBtn.addEventListener("click", async () => {
    if (entries.length && !window.confirm("Delete every log entry? This cannot be undone.")) {
      return;
    }

    const restore = setButtonBusy(clearBtn);

    try {
      await api("/api/logs", { method: "DELETE" });
      toast("success", "Log cleared");
      await load();
    } catch (error) {
      toast("error", "Could not clear the log", error.message);
    } finally {
      restore();
    }
  });
}

/** Bind the station and load its entries the first time it is opened. */
export function initLogsSection() {
  bind();
  onTabFirstOpen("log", load);
}
