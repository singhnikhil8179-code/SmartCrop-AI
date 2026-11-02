document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("cropForm");
  if (form) {
    form.addEventListener("submit", () => {
      const button = form.querySelector('button[type="submit"]');
      if (button) { button.disabled = true; button.textContent = "Generating recommendations..."; }
    });
  }
});
