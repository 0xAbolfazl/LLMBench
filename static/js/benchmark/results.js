// Renderers for a finished comparison: the verdict, the metric tiles, the
// per-metric bars, the summary table and the per-prompt detail.
//
// The engine answers with {models, tests: [{name, model, configuration,
// results, summary}, ...], significance: {by_model, across_models}}, so
// everything here is derived from that one shape.

import { escapeHtml, num } from "../lib/format.js";
import { indexOfName, seriesColor } from "./store.js";
import { optionChips } from "./configs.js";

// Metrics shown as tiles and as bar blocks. better tells the ranking which
// direction wins: generation rate should be high, wall time should be low.
const METRICS = [
  {
    key: "average_output_tokens_per_second",
    title: "Output speed",
    note: "Generated tokens per second",
    short: "out tok/s",
    better: "high",
    digits: 1,
  },
  {
    key: "average_prompt_tokens_per_second",
    title: "Prompt speed",
    note: "Prompt tokens processed per second",
    short: "prompt tok/s",
    better: "high",
    digits: 1,
  },
  {
    key: "average_duration_seconds",
    title: "Answer time",
    note: "Wall-clock seconds per prompt",
    short: "seconds",
    better: "low",
    digits: 2,
  },
  {
    key: "average_ttft_seconds",
    title: "First token",
    note: "Delay to the first streamed token",
    short: "ttft s",
    better: "low",
    digits: 3,
  },
];

/**
 * Pick the significance assessment the run shape cares about.
 *
 * The engine returns both halves of the verdict; a cross-model run is judged
 * across models, a single-model run within that model's configurations. The
 * single-model name comes from the job snapshot (the engine's result only
 * carries the plural models list).
 *
 * @param {object} job Finished job snapshot.
 * @returns {object|null} The relevant assessment, or null when unmeasured.
 */
function relevantSignificance(job) {
  const result = job.result || {};
  const significance = result.significance || null;

  if (!significance) {
    return null;
  }

  if (job.cross_model) {
    return significance.across_models || null;
  }

  const model = job.model || (result.models || [])[0];
  return (significance.by_model && significance.by_model[model]) || null;
}

/**
 * Derive the comparison-level TTFT average from the per-prompt rows.
 *
 * The engine times the first streamed token per prompt; the four-metric layout
 * wants it averaged per entry like the other three metrics, so the mean is
 * taken here from exactly the rows that reported one.
 *
 * @param {Array<object>} tests Tests from the result, updated in place.
 */
function deriveTtftAverages(tests) {
  tests.forEach((test) => {
    const values = (test.results || [])
      .filter(
        (row) => row.success && row.ttft_seconds !== null && row.ttft_seconds !== undefined
      )
      .map((row) => row.ttft_seconds);

    test.summary.average_ttft_seconds = values.length
      ? values.reduce((sum, value) => sum + value, 0) / values.length
      : null;
  });
}

/**
 * Format a metric as its mean with the noise band the repetitions measured.
 *
 * @param {object} summary One test's summary statistics.
 * @param {object} metric Metric descriptor.
 * @returns {string} Display text such as "12.4 ± 0.3".
 */
function numWithNoise(summary, metric) {
  const mean = num(summary[metric.key], metric.digits);

  if (summary[metric.key] === null || summary[metric.key] === undefined) {
    return mean;
  }

  const spread = summary[`${metric.key.replace(/^average_/, "")}_stddev`];

  if (spread === null || spread === undefined) {
    return mean;
  }

  return `${mean} ± ${Number(spread).toFixed(metric.digits)}`;
}

/**
 * Return the colour a test is drawn in.
 *
 * Results outlive the configuration list — the list can be edited after a run
 * — so a name that is no longer present falls back to its position.
 *
 * @param {object} test One entry of the tests array.
 * @param {number} position Zero-based position in the results.
 * @returns {string} CSS colour.
 */
function testColor(test, position) {
  const index = indexOfName(test.name);
  return seriesColor(index === -1 ? position : index);
}

/**
 * Find the best value of one metric across every test.
 *
 * @param {Array<object>} tests Tests from the result.
 * @param {object} metric Metric descriptor.
 * @returns {number|null} Best value, or null when no test reported one.
 */
function bestValue(tests, metric) {
  const values = tests
    .map((test) => test.summary[metric.key])
    .filter((value) => value !== null && value !== undefined);

  if (values.length === 0) {
    return null;
  }

  return metric.better === "high" ? Math.max(...values) : Math.min(...values);
}

/**
 * Build the metric tiles, one per metric.
 *
 * @param {Array<object>} tests Tests from the result.
 * @returns {string} HTML markup.
 */
function metricTiles(tests) {
  const bests = METRICS.map((metric) => bestValue(tests, metric));

  return METRICS.map((metric, metricIndex) => {
    const rows = tests
      .map((test, position) => {
        const value = test.summary[metric.key];
        const missing = value === null || value === undefined;
        const isBest = !missing && value === bests[metricIndex] && tests.length > 1;

        return `
          <div class="metric-tile${isBest ? " is-winner" : ""}" style="--series-color:${testColor(test, position)}">
            <div class="metric-tile-head">
              <span class="metric-tile-title">${escapeHtml(metric.title)}</span>
              <span class="metric-tile-name" title="${escapeHtml(test.name)}">${escapeHtml(test.name)}</span>
            </div>
            <div class="metric-tile-value${isBest ? " is-best" : ""}${missing ? " is-empty" : ""}" dir="ltr">
              ${escapeHtml(numWithNoise(test.summary, metric))}
            </div>
            <div class="metric-tile-noise">${escapeHtml(metric.note)}</div>
          </div>`;
      })
      .join("");

    return rows;
  }).join("");
}

/**
 * Build the bar blocks, one per metric.
 *
 * Bars are scaled against the best value in their own block, so each block
 * ranks its entries on its own terms and the blocks stay comparable side by
 * side.
 *
 * @param {Array<object>} tests Tests from the result.
 * @returns {string} HTML markup.
 */
function metricBars(tests) {
  return METRICS.map((metric) => {
    const values = tests
      .map((test) => test.summary[metric.key])
      .filter((value) => value !== null && value !== undefined);

    const max = values.length ? Math.max(...values) : 0;
    const min = values.length ? Math.min(...values) : 0;
    const best = bestValue(tests, metric);

    const rows = tests
      .map((test, position) => {
        const value = test.summary[metric.key];
        const missing = value === null || value === undefined;

        // For a "lower is better" metric the shortest bar would otherwise be
        // the winner, which reads backwards, so the scale is inverted: the
        // best value fills the track in both directions.
        let width = 0;

        if (!missing) {
          if (metric.better === "high") {
            width = max > 0 ? (value / max) * 100 : 0;
          } else {
            width = value > 0 ? (min / value) * 100 : 0;
          }
        }

        const isBest = !missing && value === best && tests.length > 1;

        return `
          <div class="bar-row" style="--series-color:${testColor(test, position)}">
            <span class="bar-name" dir="ltr" title="${escapeHtml(test.name)}">${escapeHtml(test.name)}</span>
            <span class="bar-value${isBest ? " is-best" : ""}${missing ? " is-empty" : ""}" dir="ltr">
              ${escapeHtml(numWithNoise(test.summary, metric))}
            </span>
            <span class="bar-track">
              <span class="bar-fill" style="width:${width.toFixed(1)}%"></span>
            </span>
          </div>`;
      })
      .join("");

    return `
      <section class="bar-block">
        <div class="bar-block-head">
          <h3 class="bar-block-title">${escapeHtml(metric.title)}</h3>
          <span class="bar-block-note">${escapeHtml(metric.note)}</span>
        </div>
        <div>${rows}</div>
      </section>`;
  }).join("");
}

/**
 * Build the summary table.
 *
 * @param {Array<object>} tests Tests from the result.
 * @returns {string} HTML markup.
 */
function summaryTable(tests) {
  const bests = METRICS.map((metric) => bestValue(tests, metric));

  const rows = tests
    .map((test, position) => {
      const failed = test.results.filter((result) => !result.success).length;

      const cells = METRICS.map((metric, index) => {
        const value = test.summary[metric.key];
        const isBest =
          value !== null && value !== undefined && value === bests[index] && tests.length > 1;
        return `<td class="is-number${isBest ? " is-best" : ""}">${escapeHtml(
          numWithNoise(test.summary, metric)
        )}</td>`;
      }).join("");

      return `
        <tr class="series-row" style="--series-color:${testColor(test, position)}">
          <td class="cell-config" dir="ltr">${escapeHtml(test.name)}</td>
          ${cells}
          <td class="is-number">${test.summary.total_output_tokens}</td>
          <td class="is-number${failed ? " is-failed" : ""}">
            ${escapeHtml(failed ? `${failed} failed` : "all passed")}
          </td>
        </tr>`;
    })
    .join("");

  const headers = METRICS.map(
    (metric) => `<th class="is-number">${escapeHtml(metric.short)}</th>`
  ).join("");

  return `
    <div class="table-wrap">
      <table class="data-table" aria-label="Result summary">
        <thead>
          <tr>
            <th>Entry</th>
            ${headers}
            <th class="is-number">out tokens</th>
            <th class="is-number">prompts</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

/**
 * Build the verdict callout naming the entry to keep.
 *
 * Output speed decides it: it is the metric a user of the model actually waits
 * on, and response time follows from it for a fixed prompt set. When the
 * comparison measured each prompt more than once, the engine's own
 * significance assessment is shown under the verdict — it says whether the gap
 * survives the noise, the difference between a finding and a coin flip.
 *
 * @param {Array<object>} tests Tests from the result.
 * @param {object|null} significance The relevant assessment.
 * @returns {string} HTML markup.
 */
function verdict(tests, significance) {
  const metric = METRICS[0];
  const best = bestValue(tests, metric);

  if (best === null) {
    return `
      <div class="verdict is-empty">
        <div>
          <div class="verdict-label">No verdict</div>
          <div class="verdict-note">No entry produced a measurable output speed.</div>
        </div>
      </div>`;
  }

  const winner = tests.find((test) => test.summary[metric.key] === best);
  const others = tests.filter((test) => test !== winner);
  const rates = others
    .map((test) => test.summary[metric.key])
    .filter((value) => value !== null && value !== undefined);

  let note = "The only entry measured.";

  if (rates.length > 0) {
    const runnerUp = Math.max(...rates);
    const gain = runnerUp > 0 ? ((best - runnerUp) / runnerUp) * 100 : 0;

    // The noise assessment overrides the raw gain: announcing "12% faster"
    // above an assessment that says the gap is within noise would answer the
    // question twice with opposite answers.
    if (significance && significance.significant === false) {
      note = "The gap is within the measured noise.";
    } else if (gain >= 0.5) {
      note = `Faster than the next entry by ${gain.toFixed(1)}%.`;
    } else {
      note = "Effectively tied with the next entry.";
    }
  }

  let assessment = "";

  if (significance) {
    const tone =
      significance.significant === true
        ? "is-good"
        : significance.significant === false
          ? "is-tied"
          : "is-unmeasured";

    const label =
      significance.significant === true
        ? "Difference is real"
        : significance.significant === false
          ? "Within noise"
          : "Not measured for noise";

    const message = significance.message || "";

    assessment = `
      <div class="verdict-assess ${tone}">
        <span class="verdict-assess-label">${escapeHtml(label)}</span>
        <span class="verdict-assess-message" dir="auto">${escapeHtml(message)}</span>
      </div>`;
  }

  return `
    <div class="verdict">
      <div>
        <div class="verdict-label">Fastest entry</div>
        <div class="verdict-name" dir="ltr">${escapeHtml(winner.name)}</div>
        <div class="verdict-note">${escapeHtml(note)}</div>
      </div>
      ${assessment}
    </div>`;
}

/**
 * Build the collapsible per-prompt detail for one test.
 *
 * @param {object} test One entry of the tests array.
 * @param {number} position Zero-based position in the results.
 * @returns {string} HTML markup.
 */
function detail(test, position) {
  const prompts = test.results
    .map((result) => {
      if (!result.success) {
        return `
          <div class="prompt-row">
            <div class="prompt-head">
              <span class="prompt-index" dir="ltr">#${result.index}</span>
              <span class="prompt-text" title="${escapeHtml(result.prompt)}">${escapeHtml(result.prompt)}</span>
              <span class="chip chip-error">failed</span>
            </div>
            <div class="prompt-error">${escapeHtml(result.error || "Unknown error")}</div>
          </div>`;
      }

      const output =
        result.response === undefined
          ? ""
          : `<pre class="output">${escapeHtml(result.response || "(empty)")}</pre>`;

      // The machine readings are optional: they are absent on a machine
      // without an NVIDIA GPU, or when a generation produced no content, and
      // are shown only when the engine actually reported them.
      const measurements = [
        { key: "ttft_seconds", digits: 3, stat: "s ttft" },
        { key: "vram_used_mb", digits: 0, stat: "MB vram" },
        { key: "gpu_temperature_c", digits: 1, stat: "°C" },
        { key: "gpu_clock_mhz", digits: 0, stat: "MHz" },
      ]
        .map((reading) => ({ ...reading, value: result[reading.key] }))
        .filter((reading) => reading.value !== null && reading.value !== undefined);

      const measurementStats = measurements
        .map(
          (reading) => `
            <span class="stat"><b dir="ltr">${num(
              reading.value,
              reading.digits
            )}</b> ${escapeHtml(reading.stat)}</span>`
        )
        .join("");

      const measurementBlock = measurementStats
        ? `<div class="prompt-machine">${measurementStats}</div>`
        : "";

      return `
        <div class="prompt-row">
          <div class="prompt-head">
            <span class="prompt-index" dir="ltr">#${result.index}</span>
            <span class="prompt-text" title="${escapeHtml(result.prompt)}">${escapeHtml(result.prompt)}</span>
          </div>
          <div class="prompt-stats">
            <span class="stat"><b dir="ltr">${num(result.duration_seconds, 2)}</b> s</span>
            <span class="stat"><b dir="ltr">${num(result.output_tokens_per_second, 1)}</b> out tok/s</span>
            <span class="stat"><b dir="ltr">${num(result.prompt_tokens_per_second, 1)}</b> prompt tok/s</span>
            <span class="stat"><b dir="ltr">${result.output_tokens}</b> out tokens</span>
            <span class="stat"><b dir="ltr">${result.prompt_tokens}</b> prompt tokens</span>
          </div>
          ${measurementBlock}
          ${output}
        </div>`;
    })
    .join("");

  const failed = test.results.filter((result) => !result.success).length;
  const meta =
    `${test.results.length} prompt${test.results.length > 1 ? "s" : ""}` +
    (failed ? ` (${failed} failed)` : "");

  return `
    <article class="detail" style="--series-color:${testColor(test, position)}">
      <button type="button" class="detail-head" aria-expanded="false">
        <span class="detail-caret" aria-hidden="true">▶</span>
        <span class="detail-name" dir="ltr">${escapeHtml(test.name)}</span>
        <span class="detail-meta">${escapeHtml(meta)}</span>
      </button>
      <div class="detail-body">
        <div class="detail-options">${optionChips(test.configuration)}</div>
        ${prompts}
      </div>
    </article>`;
}

/**
 * Build the SVG output-speed chart.
 *
 * One bar per entry, longest at the top, each carrying a band as wide as its
 * measured noise — two bars whose bands overlap are saying the same speed.
 * The winner's bar gets full strength; the rest render at reduced opacity so
 * the eye lands on the finding first.
 *
 * @param {Array<object>} tests Tests from the result.
 * @returns {string} SVG markup, or "" when nothing measurable was reported.
 */
function outputSpeedChart(tests) {
  const measured = tests
    .map((test, position) => ({
      name: test.name,
      rate: test.summary.average_output_tokens_per_second,
      noise: test.summary.output_tokens_per_second_stddev,
      position,
    }))
    .filter((entry) => entry.rate !== null && entry.rate !== undefined);

  if (measured.length === 0) {
    return "";
  }

  measured.sort((a, b) => b.rate - a.rate);

  const best = measured[0].rate;
  const scaleMax = Math.max(
    best +
      measured.reduce((max, entry) => {
        const noise = entry.noise ?? 0;
        return Math.max(max, entry.rate + noise);
      }, 0),
    best * 1.05
  ) || 1;

  const plotX = 150;
  const plotWidth = 380;
  const rowHeight = 42;
  const barHeight = 20;
  const width = 640;
  const height = measured.length * rowHeight + 30;

  const scale = (value) => (value / scaleMax) * plotWidth;

  const rows = measured
    .map((entry, index) => {
      const y = index * rowHeight + 14;
      const barWidth = Math.max(scale(entry.rate), 2);
      const color = seriesColor(
        indexOfName(entry.name) === -1 ? entry.position : indexOfName(entry.name)
      );
      const isWinner = index === 0;
      const opacity = isWinner ? 1 : 0.5;

      // The noise range spans one stddev either way around the mean and is
      // drawn as a translucent red overlay across the bar plus two tick marks
      // at its ends — unmistakable at a glance.
      const noise = entry.noise ?? 0;
      const bandStart = scale(Math.max(entry.rate - noise, 0));
      const bandEnd = scale(entry.rate + noise);
      const bandWidth = Math.max(bandEnd - bandStart, 1);
      const ticks =
        noise > 0
          ? `<line x1="${(plotX + bandStart).toFixed(1)}" y1="${y}" x2="${(plotX + bandStart).toFixed(1)}" y2="${y + barHeight}"
               stroke="var(--red)" stroke-width="1.5"></line>
             <line x1="${(plotX + bandEnd).toFixed(1)}" y1="${y}" x2="${(plotX + bandEnd).toFixed(1)}" y2="${y + barHeight}"
               stroke="var(--red)" stroke-width="1.5"></line>`
          : "";

      const valueText = Number(entry.rate).toFixed(1);
      const valueX = plotX + barWidth + 8;

      return `
        <text x="0" y="${y + barHeight / 2 + 4}" class="chart-label"
              title="${escapeHtml(entry.name)}">${
        escapeHtml(entry.name.length > 20 ? `${entry.name.slice(0, 19)}…` : entry.name)
      }</text>
        <rect x="${plotX}" y="${y}" width="${plotWidth}" height="${barHeight}"
              rx="2" fill="none" stroke="var(--border)"></rect>
        <rect x="${plotX}" y="${y}" width="${barWidth.toFixed(1)}" height="${barHeight}"
              rx="2" fill="${color}" opacity="${opacity}"></rect>
        ${
          noise > 0
            ? `<rect x="${(plotX + bandStart).toFixed(1)}" y="${y}" width="${bandWidth.toFixed(1)}"
                 height="${barHeight}" fill="var(--red)" opacity="0.25"></rect>
           ${ticks}`
            : ""
        }
        <text x="${Math.min(valueX, width - 8)}" y="${y + barHeight / 2 + 4}"
              class="chart-value${isWinner ? " is-winner" : ""}">${valueText}</text>`;
    })
    .join("");

  return `
    <figure class="chart">
      <div class="chart-title">Output speed · tok/s</div>
      <svg viewBox="0 0 ${width} ${height}" role="img"
           aria-label="Output speed per entry, with noise bands">
        ${rows}
      </svg>
      <span class="chart-caption">
        Bars are generation speed; the red band is the measured run-to-run noise.
      </span>
    </figure>`;
}

/** Build the one-line sub-head: prompt count, wall time, finish time, reps. */
function resultsSub(job, reps) {
  const parts = [
    `${job.prompts.length} prompt${job.prompts.length > 1 ? "s" : ""}`,
    `${num(job.elapsed_seconds, 1)} s total`,
  ];

  if (job.finished_at) {
    parts.push(`finished ${job.finished_at}`);
  }

  if (reps > 1) {
    parts.push(`${reps}x per prompt`);
  }

  return parts.join(" · ");
}

/**
 * Paint a finished comparison.
 *
 * Both comparison kinds land here: a single-model run (several configurations)
 * and a cross-model one (several models, one shared configuration, or a
 * tournament). The engine's result shape is the same for the two, so the only
 * real difference is which significance half the verdict reads.
 *
 * @param {HTMLElement} el Container element.
 * @param {object} job Finished job snapshot from the status endpoint.
 */
export function renderResults(el, job) {
  const result = job.result || {};
  const tests = result.tests || [];

  if (tests.length === 0) {
    el.innerHTML = `<div class="empty-state">No results were produced.</div>`;
    return;
  }

  deriveTtftAverages(tests);

  const significance = relevantSignificance(job);
  const reps = job.repetitions || 1;
  const head = job.cross_model
    ? `Compared ${job.models.length} model${job.models.length > 1 ? "s" : ""} · ${tests.length} entr${tests.length > 1 ? "ies" : "y"}`
    : `${job.model} · ${tests.length} configuration${tests.length > 1 ? "s" : ""}`;

  el.innerHTML = `
    <div class="results">
      <div class="results-head">
        <div>
          <div class="results-title">${escapeHtml(head)}</div>
          <div class="results-sub">${escapeHtml(resultsSub(job, reps))}</div>
        </div>
        <div class="results-tools">
          ${job.history_id ? `<span class="chip chip-saved">saved to history</span>` : ""}
          <div class="btn-row">
            <button type="button" class="btn btn-sm" data-results-download="csv"
                    title="Download the summary table as CSV">CSV</button>
            <button type="button" class="btn btn-sm" data-results-download="json"
                    title="Download the full result as JSON">JSON</button>
            <button type="button" class="btn btn-sm btn-danger-ghost" data-results-discard>Discard</button>
          </div>
        </div>
      </div>

      ${verdict(tests, significance)}

      <div class="metric-tiles">${metricTiles(tests)}</div>

      <div class="bars">${metricBars(tests)}</div>

      ${outputSpeedChart(tests)}

      ${summaryTable(tests)}

      <div class="details">${tests.map((test, position) => detail(test, position)).join("")}</div>
    </div>`;

  el.querySelectorAll(".detail-head").forEach((headBtn) => {
    headBtn.addEventListener("click", () => {
      const article = headBtn.closest(".detail");
      const open = article.classList.toggle("is-open");
      headBtn.setAttribute("aria-expanded", open ? "true" : "false");
    });
  });
}
