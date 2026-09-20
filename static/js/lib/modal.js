// A single modal host shared by the configuration editor and the import
// drawer. The host is in the shell so a modal can open from any station; the
// content is built by the caller, which keeps this module ignorant of what is
// being edited.

const host = document.getElementById("modal-host");

let openHandle = null;

/**
 * Open a modal with a title, a body and an optional footer.
 *
 * The body and footer are HTML strings the caller built (already escaped).
 * Only one modal is open at a time; opening one closes the previous. The
 * handle carries a close() that detaches the Escape listener and runs the
 * caller's closing hook, if it registered one.
 *
 * @param {string} title Modal heading.
 * @param {string} bodyHtml Modal body markup.
 * @param {string} [footHtml] Footer markup; omitted for a borderless foot.
 * @returns {{el: HTMLElement, close: Function, onClose: Function}} Handle.
 */
export function openModal(title, bodyHtml, footHtml = null) {
  close();

  host.innerHTML = `
    <div class="modal-backdrop" data-modal-backdrop></div>
    <div class="modal" role="dialog" aria-modal="true">
      <div class="modal-head">
        <h3 class="modal-title"></h3>
        <button type="button" class="modal-close" data-modal-close aria-label="Close dialog">✕</button>
      </div>
      <div class="modal-body">${bodyHtml}</div>
      ${footHtml ? `<div class="modal-foot">${footHtml}</div>` : ""}
    </div>`;

  host.hidden = false;

  const el = host.querySelector(".modal");
  const titleEl = host.querySelector(".modal-title");
  titleEl.textContent = title;
  titleEl.setAttribute("aria-label", title);

  let closeCallback = null;

  const handle = { el, closed: false };

  handle.close = () => {
    if (handle.closed) {
      return;
    }
    handle.closed = true;

    host.hidden = true;
    host.innerHTML = "";

    if (openHandle === handle) {
      openHandle = null;
    }

    document.removeEventListener("keydown", onKey, true);

    if (closeCallback) {
      closeCallback();
    }
  };

  const onKey = (event) => {
    if (event.key === "Escape") {
      handle.close();
    }
  };

  document.addEventListener("keydown", onKey, true);
  host.querySelector("[data-modal-backdrop]").addEventListener("click", handle.close);
  host.querySelector("[data-modal-close]").addEventListener("click", handle.close);

  // Register a hook that runs once the modal has closed.
  handle.onClose = (callback) => {
    closeCallback = callback;
  };

  openHandle = handle;
  return handle;
}

/**
 * Close the open modal, if any.
 */
export function close() {
  if (!openHandle) {
    return;
  }

  const handle = openHandle;
  handle.close();
}
