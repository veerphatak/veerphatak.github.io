/* ------------------------------------------------------------------
   Interactive lap time and race strategy simulators.

   The lap solver here is a direct port of the Python model, running on
   circuit geometry derived from real position telemetry. Change the car,
   and the lap is re-solved in the browser.
   ------------------------------------------------------------------ */
(() => {
  "use strict";

  const NAVY = "#58a6ff", RED = "#ff6b5a", MUTED = "#8b949e", GRID = "#233041";
  const RHO = 1.20, G = 9.81;

  /* ---------- small canvas helper ---------- */
  function setupCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const rect = cv.getBoundingClientRect();
    cv.width = rect.width * dpr;
    cv.height = rect.height * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: rect.width, h: rect.height };
  }

  function axes(ctx, w, h, pad, xr, yr, xlabel, ylabel) {
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = GRID; ctx.lineWidth = 1;
    ctx.fillStyle = MUTED; ctx.font = "11px system-ui, sans-serif";
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (h - pad.t - pad.b) * i / 4;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      const val = yr[1] - (yr[1] - yr[0]) * i / 4;
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(Math.round(val), pad.l - 6, y);
    }
    for (let i = 0; i <= 5; i++) {
      const x = pad.l + (w - pad.l - pad.r) * i / 5;
      const val = xr[0] + (xr[1] - xr[0]) * i / 5;
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(Math.round(val), x, h - pad.b + 6);
    }
    ctx.textAlign = "center";
    ctx.fillText(xlabel, pad.l + (w - pad.l - pad.r) / 2, h - 14);
    ctx.save();
    ctx.translate(12, pad.t + (h - pad.t - pad.b) / 2);
    ctx.rotate(-Math.PI / 2); ctx.textBaseline = "top";
    ctx.fillText(ylabel, 0, 0);
    ctx.restore();
  }

  const sx = (v, xr, pad, w) => pad.l + (v - xr[0]) / (xr[1] - xr[0]) * (w - pad.l - pad.r);
  const sy = (v, yr, pad, h) => pad.t + (yr[1] - v) / (yr[1] - yr[0]) * (h - pad.t - pad.b);

  function line(ctx, xs, ys, xr, yr, pad, w, h, colour, width, dash) {
    ctx.save(); ctx.strokeStyle = colour; ctx.lineWidth = width;
    ctx.setLineDash(dash || []); ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const X = sx(xs[i], xr, pad, w), Y = sy(ys[i], yr, pad, h);
      i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
    }
    ctx.stroke(); ctx.restore();
  }

  /* =================================================================
     1. LAP SIMULATOR
     ================================================================= */
  const lapSim = {
    data: null,

    solve(p) {
      const R = this.data.radius, drs = this.data.drs, ds = this.data.ds;
      const n = R.length;
      const aeroK = p.mu * 0.5 * RHO * p.clA / p.mass;

      // cornering limit: grip rises with speed because downforce does
      const vc = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const r = R[i];
        if (r === null) { vc[i] = p.vMax; continue; }
        const bracket = 1 / r - aeroK;
        vc[i] = bracket > 0 ? Math.min(Math.sqrt(p.mu * G / bracket), p.vMax) : p.vMax;
      }

      const drag = (v, d) => 0.5 * RHO * p.cdA * (1 - 0.12 * d) * v * v;
      const load = v => p.mass * G + 0.5 * RHO * p.clA * v * v;

      // forward pass: traction limited
      const vf = new Float64Array(n); vf[0] = vc[0];
      for (let i = 0; i < n - 1; i++) {
        const v = Math.min(vf[i], vc[i]);
        const grip = p.mu * load(v);
        const lat = R[i] === null ? 0 : p.mass * v * v / R[i];
        const rem = Math.sqrt(Math.max(grip * grip - lat * lat, 0));
        const f = Math.min(p.power / Math.max(v, 1), rem) - drag(v, drs[i]);
        vf[i + 1] = Math.min(Math.sqrt(Math.max(v * v + 2 * (f / p.mass) * ds, 1)), p.vMax);
      }

      // backward pass: braking limited
      const vb = new Float64Array(n); vb[n - 1] = vc[n - 1];
      for (let i = n - 1; i > 0; i--) {
        const v = Math.min(vb[i], vc[i]);
        const grip = p.mu * load(v);
        const lat = R[i] === null ? 0 : p.mass * v * v / R[i];
        const rem = Math.sqrt(Math.max(grip * grip - lat * lat, 0));
        const f = Math.min(rem, p.brakeG * p.mass * G) + drag(v, drs[i]);
        vb[i - 1] = Math.min(Math.sqrt(Math.max(v * v + 2 * (f / p.mass) * ds, 1)), p.vMax);
      }

      const v = new Float64Array(n);
      let t = 0;
      for (let i = 0; i < n; i++) {
        v[i] = Math.min(vf[i], vb[i], vc[i]);
        t += ds / v[i];
      }
      return { speed: v, lapTime: t };
    },

    params() {
      const g = id => parseFloat(document.getElementById(id).value);
      return {
        mu: g("mu"), clA: g("clA"), cdA: g("cdA"), mass: g("mass"),
        power: g("power") * 1000, brakeG: g("brakeG"),
        vMax: this.data.fitted.vMax
      };
    },

    render() {
      const p = this.params();
      const { speed, lapTime } = this.solve(p);
      const d = this.data.distance, real = this.data.speedReal;

      // stats
      let se = 0, vmax = 0, vmin = 1e9;
      for (let i = 0; i < speed.length; i++) {
        const kmh = speed[i] * 3.6;
        se += (kmh - real[i]) ** 2;
        if (kmh > vmax) vmax = kmh;
        if (kmh < vmin) vmin = kmh;
      }
      const rmse = Math.sqrt(se / speed.length);
      const delta = lapTime - this.data.realLapTime;

      document.getElementById("lapTime").textContent = lapTime.toFixed(3) + " s";
      const dEl = document.getElementById("lapDelta");
      dEl.textContent = (delta >= 0 ? "+" : "") + delta.toFixed(3) + " s vs real";
      dEl.style.color = Math.abs(delta) < 1 ? "#3fb950" : (delta > 0 ? "#ff6b5a" : NAVY);
      document.getElementById("lapRmse").textContent = rmse.toFixed(1) + " km/h";
      document.getElementById("lapVmax").textContent = Math.round(vmax) + " km/h";
      document.getElementById("lapVmin").textContent = Math.round(vmin) + " km/h";

      // chart
      const cv = document.getElementById("lapChart");
      const { ctx, w, h } = setupCanvas(cv);
      const pad = { l: 46, r: 12, t: 14, b: 40 };
      const xr = [0, d[d.length - 1]], yr = [0, 350];
      axes(ctx, w, h, pad, xr, yr, "Distance around lap (m)", "Speed (km/h)");

      // corner shading
      ctx.save(); ctx.fillStyle = "rgba(139,148,158,0.07)";
      for (let i = 0; i < this.data.radius.length; i++) {
        const r = this.data.radius[i];
        if (r !== null && r < 250) {
          const X = sx(d[i], xr, pad, w);
          ctx.fillRect(X, pad.t, Math.max(1, (w - pad.l - pad.r) / d.length + 0.6), h - pad.t - pad.b);
        }
      }
      ctx.restore();

      line(ctx, d, real, xr, yr, pad, w, h, NAVY, 2);
      line(ctx, d, Array.from(speed, v => v * 3.6), xr, yr, pad, w, h, RED, 1.8, [6, 4]);

      // legend
      ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillStyle = NAVY; ctx.fillRect(w - 190, pad.t + 8, 14, 3);
      ctx.fillText("Measured (real pole lap)", w - 170, pad.t + 9);
      ctx.fillStyle = RED; ctx.fillRect(w - 190, pad.t + 26, 14, 3);
      ctx.fillText("Your car", w - 170, pad.t + 27);
    },

    init(data) {
      this.data = data;
      document.getElementById("simCircuit").textContent =
        `${data.circuit}, ${data.driver} (${data.team}), ${data.realLapTime.toFixed(3)} s`;
      const ids = ["mu", "clA", "cdA", "mass", "power", "brakeG"];
      ids.forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener("input", () => {
          document.getElementById(id + "Val").textContent = el.value;
          this.render();
        });
        document.getElementById(id + "Val").textContent = el.value;
      });
      document.getElementById("resetSim").addEventListener("click", () => {
        const f = data.fitted;
        const set = (id, v) => {
          document.getElementById(id).value = v;
          document.getElementById(id + "Val").textContent = v;
        };
        set("mu", f.mu); set("clA", f.clA); set("cdA", f.cdA);
        set("mass", f.mass); set("power", Math.round(f.power / 1000)); set("brakeG", f.brakeG);
        this.render();
      });
      window.addEventListener("resize", () => this.render());
      this.render();
    }
  };

  /* =================================================================
     2. STRATEGY CALCULATOR
     ================================================================= */
  const strategy = {
    read() {
      const g = id => parseFloat(document.getElementById(id).value);
      return {
        laps: g("raceLaps"), pit: g("pitLoss"), fuel: -Math.abs(g("fuelEffect")),
        base: { S: g("baseS"), M: g("baseM"), H: g("baseH") },
        deg: { S: g("degS"), M: g("degM"), H: g("degH") }
      };
    },

    stintTime(p, c, n, startLap) {
      let t = 0;
      for (let a = 1; a <= n; a++) t += p.base[c] + p.deg[c] * a + p.fuel * (startLap + a - 1);
      return t;
    },

    best(p) {
      const comps = ["S", "M", "H"], MIN = 6;
      let best = null;
      const window1 = [], window2 = [];

      // one stop
      for (let n1 = MIN; n1 <= p.laps - MIN; n1++) {
        let cell = Infinity;
        for (const a of comps) for (const b of comps) {
          if (a === b) continue;                    // two compounds required
          const t = this.stintTime(p, a, n1, 1) + this.stintTime(p, b, p.laps - n1, n1 + 1) + p.pit;
          if (t < cell) cell = t;
          if (!best || t < best.time) best = { time: t, stops: 1, plan: [[a, n1], [b, p.laps - n1]] };
        }
        window1.push([n1, cell]);
      }

      // two stops
      for (let n1 = MIN; n1 <= p.laps - 2 * MIN; n1++) {
        let cell = Infinity;
        for (let n2 = MIN; n2 <= p.laps - n1 - MIN; n2++) {
          const n3 = p.laps - n1 - n2;
          for (const a of comps) for (const b of comps) for (const c of comps) {
            if (new Set([a, b, c]).size < 2) continue;
            const t = this.stintTime(p, a, n1, 1)
                    + this.stintTime(p, b, n2, n1 + 1)
                    + this.stintTime(p, c, n3, n1 + n2 + 1) + 2 * p.pit;
            if (t < cell) cell = t;
            if (t < best.time) best = { time: t, stops: 2, plan: [[a, n1], [b, n2], [c, n3]] };
          }
        }
        window2.push([n1, cell]);
      }
      return { best, window1, window2 };
    },

    render() {
      const p = this.read();
      const { best, window1, window2 } = this.best(p);

      const mm = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
      document.getElementById("stratTime").textContent = mm(best.time);
      document.getElementById("stratStops").textContent = best.stops + (best.stops === 1 ? " stop" : " stops");
      const names = { S: "Soft", M: "Medium", H: "Hard" };
      document.getElementById("stratPlan").textContent =
        best.plan.map(([c, n]) => `${names[c]} ${n}`).join("  →  ");

      const cv = document.getElementById("stratChart");
      const { ctx, w, h } = setupCanvas(cv);
      const pad = { l: 52, r: 12, t: 14, b: 40 };
      const ref = best.time;
      const all = window1.concat(window2).map(d => d[1] - ref).filter(v => isFinite(v));
      const yr = [0, Math.min(60, Math.max(10, Math.ceil(Math.min(...all.map(v => v)) + 40)))];
      const xr = [0, p.laps];
      axes(ctx, w, h, pad, xr, yr, "Lap of the first pit stop", "Time lost vs best (s)");

      const draw = (win, colour) => {
        const xs = win.map(d => d[0]), ys = win.map(d => Math.min(d[1] - ref, yr[1]));
        line(ctx, xs, ys, xr, yr, pad, w, h, colour, 2.2);
      };
      draw(window1, MUTED);
      draw(window2, "#3fb950");

      ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillStyle = MUTED; ctx.fillRect(w - 150, pad.t + 8, 14, 3);
      ctx.fillText("One stop", w - 130, pad.t + 9);
      ctx.fillStyle = "#3fb950"; ctx.fillRect(w - 150, pad.t + 26, 14, 3);
      ctx.fillText("Two stops", w - 130, pad.t + 27);
    },

    init() {
      ["raceLaps", "pitLoss", "fuelEffect", "baseS", "baseM", "baseH", "degS", "degM", "degH"]
        .forEach(id => {
          const el = document.getElementById(id);
          if (!el) return;
          el.addEventListener("input", () => {
            document.getElementById(id + "Val").textContent = el.value;
            this.render();
          });
          document.getElementById(id + "Val").textContent = el.value;
        });
      window.addEventListener("resize", () => this.render());
      this.render();
    }
  };

  /* ---------- boot ---------- */
  document.addEventListener("DOMContentLoaded", () => {
    if (document.getElementById("lapChart")) {
      fetch("data/bahrain-lap.json")
        .then(r => r.json())
        .then(d => lapSim.init(d))
        .catch(() => {
          document.getElementById("simCircuit").textContent =
            "Could not load the circuit data.";
        });
    }
    if (document.getElementById("stratChart")) strategy.init();
  });
})();
