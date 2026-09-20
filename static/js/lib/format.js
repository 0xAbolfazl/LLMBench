// Small formatting and HTML-building helpers shared by every panel.

/**
 * Escape a value for interpolation into an HTML string.
 *
 * Model tags and prompt text come straight from the server or the user, so
 * nothing reaches the DOM without passing through here.
 *
 * @param {*} value Value to escape.
 * @returns {string} Escaped text.
 */
export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
  });
}

/**
 * Format a number for display, or an em dash when it is missing.
 *
 * A metric is null whenever Ollama reported no timing for it, which is not the
 * same as zero and must not be ranked as if it were.
 *
 * @param {*} value Metric value.
 * @param {number} digits Decimal places.
 * @returns {string} Display text.
 */
export function num(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return Number(value).toFixed(digits);
}

/**
 * Format a byte count as a short human size.
 *
 * @param {number|null|undefined} bytes Bytes, or nothing when unreported.
 * @returns {string} Display text such as "8.2 GB", or an em dash.
 */
export function humanBytes(bytes) {
  if (bytes === null || bytes === undefined || Number.isNaN(Number(bytes))) {
    return "—";
  }

  const value = Number(bytes);
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let index = 0;

  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }

  const digits = index === 0 ? 0 : size < 10 ? 1 : 0;

  return `${size.toFixed(digits)} ${units[index]}`;
}

/**
 * Format an epoch-seconds timestamp as a local date-time string.
 *
 * @param {number|null|undefined} epochSeconds Epoch seconds, or nothing.
 * @returns {string} Local date-time, or an em dash.
 */
export function humanDateTime(epochSeconds) {
  if (epochSeconds === null || epochSeconds === undefined) {
    return "—";
  }

  const date = new Date(Number(epochSeconds) * 1000);

  if (Number.isNaN(date.getTime())) {
    return "—";
  }

  return date.toLocaleString();
}

/** Show an inline alert bar.
 *
 * @param {HTMLElement} el Alert element.
 * @param {string} message Text to show.
 */
export function showAlert(el, message) {
  el.textContent = message;
  el.classList.remove("is-hidden");
}

/** Hide an inline alert bar.
 *
 * @param {HTMLElement} el Alert element.
 */
export function hideAlert(el) {
  el.classList.add("is-hidden");
}
