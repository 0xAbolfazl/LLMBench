// LLMBench — dashboard entrypoint.
//
// Bind the tab rail and every station, then let the benchmark workbench load
// the engine registry and its schema. Once its server-state listener is
// registered, fetch the selected engine's status and model catalog: the
// command-bar chip, the setup model list, the station badges and the engine
// station all paint from that first refresh.

import { initTabs } from "./tabs.js";
import { initBenchmarkPanel } from "./benchmark/panel.js";
import { initHistoryPanel } from "./benchmark/history.js";
import { initLogsSection } from "./logs.js";
import { initEnginePanel } from "./engine/panel.js";
import { refreshServer, bindEngineSwitching } from "./lib/server.js";
import { initCollapsibleBlocks } from "./lib/collapse.js";

initTabs();
initCollapsibleBlocks();
initHistoryPanel();
initLogsSection();
initEnginePanel();

// initBenchmarkPanel registers its onServerChange listener only after an
// awaited schema fetch, so the first server refresh must wait for it: an
// earlier notify would miss the setup model list and the readiness display.
initBenchmarkPanel().then(() => {
  bindEngineSwitching(refreshServer);
  refreshServer();
});
