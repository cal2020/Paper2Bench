// paper2bench docs — minimal interactivity

document.addEventListener("click", (e) => {
  // Copy buttons
  const btn = e.target.closest(".copy-btn");
  if (btn) {
    const target = btn.parentElement.querySelector("pre, code");
    if (target) {
      navigator.clipboard.writeText(target.innerText).then(() => {
        const original = btn.textContent;
        btn.textContent = "copied";
        btn.classList.add("copied");
        setTimeout(() => {
          btn.textContent = original;
          btn.classList.remove("copied");
        }, 1400);
      });
    }
    return;
  }

  // Tabs (example detail page)
  const tab = e.target.closest(".tab");
  if (tab) {
    const tabRow = tab.parentElement;
    const tabsRoot = tabRow.parentElement;
    tabRow.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const target = tab.dataset.tab;
    tabsRoot.querySelectorAll(".tab-panel").forEach((p) => {
      p.classList.toggle("active", p.dataset.tab === target);
    });
  }
});
