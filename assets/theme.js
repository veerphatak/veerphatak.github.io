/* Light / dark theme.

   Dark is the default and the site's own look. Light exists for people who
   print a page, paste a screenshot into a document, or simply read better on
   a white surface. It is chosen with the button in the nav and remembered on
   this device; the operating system setting is deliberately not followed, so
   a first-time visitor always sees the site as designed.

   This file is loaded in <head>, before the stylesheet, so the saved choice
   is applied before the first paint and the page never flashes dark first. */
(() => {
  "use strict";
  const KEY = "theme";
  const read = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
  const write = v => { try { localStorage.setItem(KEY, v); } catch (e) { /* private mode */ } };

  const apply = theme => {
    if (theme === "light") document.documentElement.setAttribute("data-theme", "light");
    else document.documentElement.removeAttribute("data-theme");
  };
  apply(read());

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.querySelector(".theme-toggle");
    if (!btn) return;
    const isLight = () => document.documentElement.getAttribute("data-theme") === "light";
    const label = () => {
      btn.setAttribute("aria-pressed", String(isLight()));
      btn.title = isLight() ? "Switch to the dark theme" : "Switch to the light theme";
    };
    label();
    btn.addEventListener("click", () => {
      const next = isLight() ? "dark" : "light";
      write(next);
      // The canvas charts read their colours from the stylesheet once, when
      // their script starts, so a page that has them is reloaded to redraw
      // in the new palette. Everything else restyles in place.
      if (document.querySelector("canvas")) { location.reload(); return; }
      apply(next);
      label();
    });
  });
})();
