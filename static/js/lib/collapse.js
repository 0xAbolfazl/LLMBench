// Collapsible sections: the setup station's Models, Prompts, and
// Configurations blocks get a chevron in their header that folds the body
// away without unmounting it, so every input keeps its state while hidden.

/**
 * Make every `.is-collapsible .block-toggle` fold its section's body.
 *
 * The collapsed marker lives on the section, so CSS hides the body and the
 * header's bottom border together; `aria-expanded` stays the source of truth.
 */
export function initCollapsibleBlocks() {
  document.querySelectorAll(".is-collapsible .block-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      const section = btn.closest(".block");
      const collapsed = section.classList.toggle("is-collapsed");
      btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
      btn.title = collapsed ? "Expand section" : "Collapse section";
    });
  });
}
