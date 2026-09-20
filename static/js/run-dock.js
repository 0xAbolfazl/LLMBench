// The run dock: a mission-control strip along the bottom edge that appears
// while a benchmark is in flight and stays visible from every station.
//
// It is fed the job snapshot by the setup station's poller; the elapsed clock
// is interpolated locally on a one-second tick so it reads smooth between the
// slow status polls, and the abort button posts to the cancel endpoint.

import { postJson } from "./lib/api.js";
import { escapeHtml } from "./lib/format.js";
import { setButtonBusy, toast } from "./lib/toast.js";

const dockEl = document.getElementById("run-dock");
const fillEl = document.getElementById("dock-fill");
const stepEl = document.getElementById("dock-step");
const detailEl = document.getElementById("dock-detail");
const clockEl = document.getElementById("dock-clock");
const percentEl = document.getElementById("dock-percent");
const cancelBtn = document.getElementById("btn-dock-cancel");
const setupBadge = document.getElementById("badge-setup");

let clockTimer = null;
let startedAtMs = null;
let elapsedAtShow = 0;
let abortToast = null;

/** Whether the dock is currently showing a run. */
export function dockIsVisible() {
  return !dockEl.classList.contains("is-hidden");
}

/**
 * Build the one-line step description for the dock.
 *
 * @param {object} job Job snapshot.
 * @param {object} progress Progress dict on the job.
 * @returns {string}
 */
function stepText(job, progress) {
  const name = job.cross_model ? progress.model : progress.configuration;
  const total = job.cross_model ? progress.model_count : progress.configuration_count;
  const which = job.cross_model ? progress.model_index : progress.configuration_index;

  if (!name) {
    return "warming up…";
  }

  return `${name} (${which}/${total}) · prompt ${progress.prompt_index}/${
    progress.prompt_count
  } · rep ${progress.repetition || 1}/${progress.repetition_count}`;
}

/** Tick the elapsed clock once per second while the dock is visible. */
function tickClock() {
  if (!startedAtMs) {
    return;
  }

  const totalSeconds = elapsedAtShow + (Date.now() - startedAtMs) / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);

  clockEl.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function startClock(baseSeconds) {
  window.clearInterval(clockTimer);
  startedAtMs = Date.now();
  elapsedAtShow = baseSeconds || 0;
  tickClock();
  clockTimer = window.setInterval(tickClock, 1000);
}

/**
 * Show the dock for a running job.
 *
 * @param {object} job Job snapshot.
 */
export function showDock(job) {
  dockEl.classList.remove("is-hidden");
  paintDock(job);
  startClock(job.elapsed_seconds);

  if (setupBadge) {
    setupBadge.textContent = "run";
    setupBadge.classList.add("is-live");
  }
}

/**
 * Repaint the dock from a fresh job snapshot.
 *
 * @param {object} job Job snapshot.
 */
export function paintDock(job) {
  const progress = job.progress || null;
  const percent = progress ? progress.percent : 0;

  fillEl.style.width = `${percent}%`;
  percentEl.textContent = `${Math.round(percent)}%`;

  if (progress && progress.phase !== "starting") {
    stepEl.textContent = stepText(job, progress);
    detailEl.textContent = `since ${job.started_at} · ${job.planned_runs} generations planned`;
  } else {
    stepEl.textContent = "starting…";
    detailEl.textContent = `since ${job.started_at}`;
  }
}

/** Hide the dock and reset its clock. */
export function hideDock() {
  window.clearInterval(clockTimer);
  clockTimer = null;
  startedAtMs = null;
  elapsedAtShow = 0;

  // The abort notice is a pending toast; once the dock is gone the run is
  // over, so it must not linger.
  if (abortToast) {
    abortToast.dismiss();
    abortToast = null;
  }
  dockEl.classList.add("is-hidden");
  clockEl.textContent = "00:00";
  fillEl.style.width = "0%";
  percentEl.textContent = "0%";

  if (setupBadge) {
    setupBadge.textContent = "";
    setupBadge.classList.remove("is-live");
  }
}

/** Bind the abort button. */
export function initRunDock() {
  cancelBtn.addEventListener("click", async () => {
    const restore = setButtonBusy(cancelBtn);

    try {
      await postJson("/api/benchmark/cancel", {});
      abortToast = toast("pending", "Aborting…", "Stopping at the next safe point.");
    } catch (error) {
      toast("error", "Could not cancel", error.message);
    } finally {
      restore();
    }
  });
}
