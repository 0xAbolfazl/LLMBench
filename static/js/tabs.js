// Tab rail: switch between the four stations.
//
// Every station renders its shell immediately but loads its data lazily, the
// first time it is opened (and again on demand through its own refresh
// button). The hooks each lazy station registers here let it fetch its data on
// first open without the rail knowing what any station contains.

const TABS = ["setup", "engine", "history", "log"];

const onFirstOpen = new Map();
const opened = new Set();

/**
 * Register a hook that runs the first time a station is opened.
 *
 * @param {string} tab Tab name.
 * @param {() => void|Promise<void>} hook
 */
export function onTabFirstOpen(tab, hook) {
  onFirstOpen.set(tab, hook);
}

/**
 * Open a station: hide the others, mark the rail, and run the first-open hook
 * if the station has one and has not opened yet.
 *
 * @param {string} tab Tab name.
 */
export async function openTab(tab) {
  if (!TABS.includes(tab)) {
    return;
  }

  TABS.forEach((name) => {
    const view = document.getElementById(`view-${name}`);
    const button = document.getElementById(`tab-${name}`);

    if (name === tab) {
      view.hidden = false;
      view.classList.add("is-active");
      button.classList.add("is-active");
      button.setAttribute("aria-selected", "true");
    } else {
      view.hidden = true;
      view.classList.remove("is-active");
      button.classList.remove("is-active");
      button.setAttribute("aria-selected", "false");
    }
  });

  if (!opened.has(tab)) {
    opened.add(tab);

    const hook = onFirstOpen.get(tab);

    if (hook) {
      hook();
    }
  }
}

/**
 * Bind the rail.
 *
 * The command-bar server chip also opens the Ollama station, so it is bound
 * here too: one place that knows how a station is opened.
 */
export function initTabs() {
  document.querySelectorAll(".tabrail .tab").forEach((button) => {
    button.addEventListener("click", () => openTab(button.dataset.tab));
  });

  const chip = document.getElementById("server-chip");

  if (chip) {
    chip.addEventListener("click", () => openTab("engine"));
  }
}
