/* =================================================================
   Monte Carlo race strategy simulation.

   The deterministic calculator answers "which strategy is quickest if
   nothing goes wrong". This answers the question a strategist actually
   has on a Sunday: which strategy wins most often, and what does it
   cost when it goes wrong.

   Modelling choices that matter, and why:

   1. Tyre life and the cliff. Degradation is not linear to the end of
      a stint. Each compound has a usable life, beyond which the rate
      grows quadratically as the tyre goes off. Without this a long
      stint is only linearly slower, and the one stop looks far safer
      than it is.

   2. Two sources of tyre variation. A correlated, per-race term for
      track and car condition, which affects every strategy equally and
      therefore mostly cancels in a comparison, and an independent
      per-set term, which does not. A three stop strategy takes on one
      more set than a two stop, and so one more independent draw.

   3. Pit loss is skewed, not symmetric. A good stop cannot save five
      seconds but a bad one can cost fifteen, so the distribution is a
      normal core with an occasional slow stop and a rare disaster.

   4. Safety car hazard is not uniform. The opening laps carry a much
      higher rate than the rest of the race, which is where a large
      part of the strategic option value actually sits.

   5. Reactive pitting. A stop taken under a safety car costs a
      fraction of a green flag stop, so a strategy still holding a stop
      when the caution appears collects a discount. That option value
      is the reason a slower strategy on paper can win more races.

   Self-contained: duplicates the small canvas helpers from sim.js
   rather than reaching into that file's scope.
   ================================================================= */
(() => {
  "use strict";

  /* Three series on a dark surface. Colour is never the only cue:
     each series also has its own dash pattern, a legend entry and its
     own labelled stat tile. (The palette validator needs Node, which
     was not available here, so secondary encoding is relied on rather
     than an unverified Delta E claim.) */
  const BLUE  = "#58a6ff",
        AMBER = "#d29922",
        GREEN = "#3fb950",
        MUTED = "#8b949e",
        GRID  = "#233041",
        PANEL = "#161b22";   /* matches --bg-soft, so the legend reads as opaque */

  const SERIES = [
    { key: "one",   label: "One stop",    colour: BLUE,  dash: [6, 4] },
    { key: "two",   label: "Two stops",   colour: GREEN, dash: null   },
    { key: "three", label: "Three stops", colour: AMBER, dash: [2, 3] }
  ];

  /* Round a raw axis step up to the next readable value. The ladder is finer
     than the usual 1/2/5 so the curves are not left sitting in the bottom
     half of an over-scaled axis: a peak of 210 gives a step of 60 and a top
     of 240, not a step of 100 and a top of 400. */
  const STEPS = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
  function niceStep(x) {
    const p = Math.pow(10, Math.floor(Math.log10(x)));
    const m = x / p;
    return (STEPS.find(s => m <= s) || 10) * p;
  }

  /* ---------- canvas helpers ---------- */
  function setupCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight || 320;
    cv.width = w * dpr; cv.height = h * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function axes(ctx, w, h, pad, xr, yr, xlabel, ylabel) {
    const X = v => pad.l + (v - xr[0]) / (xr[1] - xr[0]) * (w - pad.l - pad.r);
    const Y = v => h - pad.b - (v - yr[0]) / (yr[1] - yr[0]) * (h - pad.t - pad.b);
    ctx.strokeStyle = GRID; ctx.lineWidth = 1;
    ctx.fillStyle = MUTED; ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    const dp = Math.abs(xr[1] - xr[0]) < 12 ? 1 : 0;
    for (let i = 0; i <= 5; i++) {
      const v = xr[0] + (xr[1] - xr[0]) * i / 5;
      ctx.beginPath(); ctx.moveTo(X(v), pad.t); ctx.lineTo(X(v), h - pad.b); ctx.stroke();
      ctx.fillText((v >= 0 ? "+" : "") + v.toFixed(dp), X(v), h - pad.b + 6);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const v = yr[0] + (yr[1] - yr[0]) * i / 4;
      ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke();
      ctx.fillText(v.toFixed(0), pad.l - 6, Y(v));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText(xlabel, pad.l + (w - pad.l - pad.r) / 2, h - 4);
    ctx.save();
    ctx.translate(12, pad.t + (h - pad.t - pad.b) / 2);
    ctx.rotate(-Math.PI / 2); ctx.textBaseline = "top";
    ctx.fillText(ylabel, 0, 0);
    ctx.restore();
  }

  function line(ctx, xs, ys, xr, yr, pad, w, h, colour, width, dash) {
    const X = v => pad.l + (v - xr[0]) / (xr[1] - xr[0]) * (w - pad.l - pad.r);
    const Y = v => h - pad.b - (v - yr[0]) / (yr[1] - yr[0]) * (h - pad.t - pad.b);
    ctx.save();
    ctx.strokeStyle = colour; ctx.lineWidth = width || 2;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    xs.forEach((x, i) => i ? ctx.lineTo(X(x), Y(ys[i])) : ctx.moveTo(X(x), Y(ys[i])));
    ctx.stroke();
    ctx.restore();
  }

  /* ---------- random numbers ---------- */
  let spare = null;
  function gauss() {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u, v, s2;
    do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s2 = u * u + v * v; }
    while (s2 >= 1 || s2 === 0);
    const f = Math.sqrt(-2 * Math.log(s2) / s2);
    spare = v * f;
    return u * f;
  }

  function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  const COMPS = ["S", "M", "H"];
  const NAMES = { S: "Soft", M: "Medium", H: "Hard" };

  const mc = {
    /* ---------- inputs ---------- */
    read() {
      const g = id => { const el = document.getElementById(id); return el ? parseFloat(el.value) : NaN; };
      return {
        laps: g("raceLaps"),
        fuel: -Math.abs(g("fuelEffect")),
        // means come from the deterministic calculator, so the two tools
        // can never disagree about the underlying car
        base: { S: g("baseS"), M: g("baseM"), H: g("baseH") },
        deg:  { S: g("degS"),  M: g("degM"),  H: g("degH")  },
        // usable life before the cliff, in laps
        life: { S: g("lifeS"), M: g("lifeM"), H: g("lifeH") },
        cliff: g("mcCliff"),
        // independent, per-set variation
        baseSd: { S: g("sdBaseS"), M: g("sdBaseM"), H: g("sdBaseH") },
        degSd:  { S: g("sdDegS"),  M: g("sdDegM"),  H: g("sdDegH")  },
        // correlated, per-race condition
        raceSd: g("mcRaceSd"),
        pitMean: g("mcPitMean"),
        pitSd: g("mcPitSd"),
        pSlow: g("mcSlow") / 100,
        pSc: g("mcSc") / 100,
        pScL1: g("mcScL1") / 100,
        pVsc: g("mcVsc") / 100,
        pRf: g("mcRf") / 100,
        runs: parseInt(document.getElementById("mcRuns").value, 10)
      };
    },

    /* ---------- deterministic plan search on the mean parameters ---------- */
    lapCost(p, c, age, lap) {
      let t = p.base[c] + p.deg[c] * age + p.fuel * lap;
      const over = age - p.life[c];
      if (over > 0) t += p.cliff * over * over;   // the cliff
      return t;
    },

    stintTime(p, c, n, startLap) {
      let t = 0;
      for (let a = 1; a <= n; a++) t += this.lapCost(p, c, a, startLap + a - 1);
      return t;
    },

    nominalPlans(p) {
      const MIN = 5;
      let one = null, two = null, three = null;

      for (let n1 = MIN; n1 <= p.laps - MIN; n1++) {
        for (const a of COMPS) for (const b of COMPS) {
          if (a === b) continue;                     // two compounds required
          const t = this.stintTime(p, a, n1, 1)
                  + this.stintTime(p, b, p.laps - n1, n1 + 1) + p.pitMean;
          if (!one || t < one.time) one = { time: t, plan: [[a, n1], [b, p.laps - n1]] };
        }
      }

      for (let n1 = MIN; n1 <= p.laps - 2 * MIN; n1++) {
        for (let n2 = MIN; n2 <= p.laps - n1 - MIN; n2++) {
          const n3 = p.laps - n1 - n2;
          for (const a of COMPS) for (const b of COMPS) for (const c of COMPS) {
            if (new Set([a, b, c]).size < 2) continue;
            const t = this.stintTime(p, a, n1, 1)
                    + this.stintTime(p, b, n2, n1 + 1)
                    + this.stintTime(p, c, n3, n1 + n2 + 1) + 2 * p.pitMean;
            if (!two || t < two.time) two = { time: t, plan: [[a, n1], [b, n2], [c, n3]] };
          }
        }
      }

      for (let n1 = MIN; n1 <= p.laps - 3 * MIN; n1++) {
        for (let n2 = MIN; n2 <= p.laps - n1 - 2 * MIN; n2++) {
          for (let n3 = MIN; n3 <= p.laps - n1 - n2 - MIN; n3++) {
            const n4 = p.laps - n1 - n2 - n3;
            for (const a of COMPS) for (const b of COMPS) for (const c of COMPS) for (const d of COMPS) {
              if (new Set([a, b, c, d]).size < 2) continue;
              const t = this.stintTime(p, a, n1, 1)
                      + this.stintTime(p, b, n2, n1 + 1)
                      + this.stintTime(p, c, n3, n1 + n2 + 1)
                      + this.stintTime(p, d, n4, n1 + n2 + n3 + 1) + 3 * p.pitMean;
              if (!three || t < three.time)
                three = { time: t, plan: [[a, n1], [b, n2], [c, n3], [d, n4]] };
            }
          }
        }
      }
      return { one, two, three };
    },

    /* ---------- one race: the conditions ---------- */
    sampleRace(p) {
      const n = p.laps;
      const sc = new Array(n + 2).fill(false);
      const vsc = new Array(n + 2).fill(false);
      const rf = new Array(n + 2).fill(false);

      for (let lap = 1; lap <= n; lap++) {
        if (sc[lap]) continue;
        // opening laps carry a far higher rate than the rest of the race
        const hazard = lap <= 2 ? p.pScL1 : p.pSc;
        if (Math.random() < hazard) {
          const dur = 3 + Math.floor(Math.random() * 3);
          for (let k = 0; k < dur && lap + k <= n; k++) sc[lap + k] = true;
        }
      }
      for (let lap = 1; lap <= n; lap++) {
        if (sc[lap] || vsc[lap]) continue;
        if (Math.random() < p.pVsc) {
          const dur = 1 + Math.floor(Math.random() * 2);
          for (let k = 0; k < dur && lap + k <= n; k++) if (!sc[lap + k]) vsc[lap + k] = true;
        }
      }
      if (Math.random() < p.pRf) rf[1 + Math.floor(Math.random() * n)] = true;

      // one correlated condition term for the whole race, shared by every
      // strategy, so it moves absolute times but largely cancels in the
      // head to head
      const condition = gauss() * p.raceSd;

      // pre-drawn pit losses, so each strategy's nth stop faces the same
      // luck and the comparison is not decided by the random number stream
      const pit = [];
      for (let i = 0; i < 3; i++) {
        let loss = p.pitMean + gauss() * p.pitSd;
        const r = Math.random();
        if (r < p.pSlow * 0.15) loss += 5 + Math.random() * 12;   // disaster: wheelgun, cross-threaded nut
        else if (r < p.pSlow) loss += 0.8 + Math.random() * 2.5;  // a slow but unremarkable stop
        pit.push(Math.max(0, loss));
      }
      return { sc, vsc, rf, condition, pit };
    },

    /* ---------- one race: one set of tyres per stint ---------- */
    sampleSets(p, nStints) {
      const sets = [];
      for (let i = 0; i < nStints; i++) {
        const base = {}, deg = {};
        for (const c of COMPS) {
          base[c] = p.base[c] + gauss() * p.baseSd[c];
          deg[c] = Math.max(0, p.deg[c] + gauss() * p.degSd[c]);
        }
        sets.push({ base, deg });
      }
      return sets;
    },

    /* ---------- run one plan through one sampled race ---------- */
    runPlan(plan, sets, ev, p) {
      const nStops = plan.length - 1;
      const sched = [];
      let acc = 0;
      for (let i = 0; i < nStops; i++) { acc += plan[i][1]; sched.push(acc); }

      let t = 0, ci = 0, age = 0, done = 0;
      const MIN_STINT = 4;

      for (let lap = 1; lap <= p.laps; lap++) {
        age++;
        const c = plan[ci][0];
        const set = sets[ci];
        t += set.base[c] + set.deg[c] * age + p.fuel * lap + ev.condition;
        const over = age - p.life[c];
        if (over > 0) t += p.cliff * over * over;

        if (done < nStops) {
          const lapsLeft = p.laps - lap;
          const mature = age >= 0.6 * plan[ci][1];
          const cheap = ev.rf[lap] || ev.sc[lap] || ev.vsc[lap];

          if (lap === sched[done] || (cheap && mature && lapsLeft >= MIN_STINT)) {
            let loss = ev.pit[done];
            if (ev.rf[lap]) loss = 0;             // the field is stopped anyway
            else if (ev.sc[lap]) loss *= 0.45;    // everyone is slower, so less is lost
            else if (ev.vsc[lap]) loss *= 0.70;
            t += loss;
            done++; ci++; age = 0;
            if (done < nStops) sched[done] = Math.max(lap + MIN_STINT, sched[done]);
          }
        }
      }
      return t;
    },

    /* ---------- the sweep ---------- */
    simulate(p, plans) {
      const keys = SERIES.map(s => s.key).filter(k => plans[k]);
      const times = {}; const wins = {};
      keys.forEach(k => { times[k] = []; wins[k] = 0; });

      const maxStints = Math.max(...keys.map(k => plans[k].plan.length));

      for (let i = 0; i < p.runs; i++) {
        const ev = this.sampleRace(p);
        const sets = this.sampleSets(p, maxStints);   // same sets available to all
        let bestK = null, bestT = Infinity;
        for (const k of keys) {
          const t = this.runPlan(plans[k].plan, sets, ev, p);
          times[k].push(t);
          if (t < bestT) { bestT = t; bestK = k; }
        }
        wins[bestK]++;
      }
      keys.forEach(k => times[k].sort((a, b) => a - b));
      return { times, wins, keys };
    },

    density(values, lo, hi, bins) {
      const d = new Array(bins).fill(0);
      const wdt = (hi - lo) / bins;
      for (const v of values) {
        const i = Math.floor((v - lo) / wdt);
        if (i >= 0 && i < bins) d[i]++;
      }
      // Smoothed, but left as a count of races per bin rather than
      // normalised to each series' own peak. Every strategy faces the same
      // number of races, so counts stay directly comparable between curves,
      // and a tighter distribution is allowed to peak higher instead of
      // being flattened to match the others.
      return d.map((_, i) => ((d[i - 1] || 0) + 2 * d[i] + (d[i + 1] || 0)) / 4);
    },

    /* ---------- render ---------- */
    render() {
      const p = this.read();
      if (!isFinite(p.laps)) return;

      const status = document.getElementById("mcStatus");
      const plans = this.nominalPlans(p);
      const r = this.simulate(p, plans);

      const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
      const mm = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
      const fmtPlan = pl => pl.map(([c, n]) => `${NAMES[c]} ${n}`).join("  →  ");

      // headline: which strategy wins most often
      let topK = r.keys[0];
      for (const k of r.keys) if (r.wins[k] > r.wins[topK]) topK = k;
      const topS = SERIES.find(s => s.key === topK);
      set("mcWinner", topS.label);
      set("mcWinProb", `${(r.wins[topK] / p.runs * 100).toFixed(1)}%`);

      const med = {};
      r.keys.forEach(k => { med[k] = quantile(r.times[k], 0.5); });
      const fastestMed = Math.min(...r.keys.map(k => med[k]));

      r.keys.forEach(k => {
        const s = SERIES.find(x => x.key === k);
        set(`mc_${k}_win`, `${(r.wins[k] / p.runs * 100).toFixed(1)}%`);
        set(`mc_${k}_med`, `${med[k] - fastestMed === 0 ? "" : "+"}${(med[k] - fastestMed).toFixed(1)} s`);
        set(`mc_${k}_spread`, `${(quantile(r.times[k], 0.95) - quantile(r.times[k], 0.05)).toFixed(1)} s`);
        set(`mc_${k}_worst`, `+${(quantile(r.times[k], 0.99) - fastestMed).toFixed(1)} s`);
        set(`mc_${k}_plan`, fmtPlan(plans[k].plan));
        set(`mc_${k}_label`, s.label);
      });
      set("mcMedianTime", mm(fastestMed));

      if (status) status.textContent =
        `${p.runs.toLocaleString()} races simulated. Median of the quickest strategy: ${mm(fastestMed)}.`;

      /* ---------- distribution plot ---------- */
      const cv = document.getElementById("mcChart");
      if (!cv) return;
      const { ctx, w, h } = setupCanvas(cv);
      const pad = { l: 54, r: 14, t: 16, b: 44 };

      const pooled = [];
      r.keys.forEach(k => { for (const v of r.times[k]) pooled.push(v - fastestMed); });
      pooled.sort((a, b) => a - b);
      const lo = quantile(pooled, 0.005), hi = quantile(pooled, 0.99);
      const span = Math.max(hi - lo, 4);
      const xr = [lo - span * 0.04, hi + span * 0.04];
      const BINS = 72;

      const xs = [];
      for (let i = 0; i < BINS; i++) xs.push(xr[0] + (xr[1] - xr[0]) * (i + 0.5) / BINS);

      const curves = [];
      let peak = 0;
      for (const s of SERIES) {
        if (!r.times[s.key]) continue;
        const d = this.density(r.times[s.key].map(v => v - fastestMed), xr[0], xr[1], BINS);
        for (const v of d) if (v > peak) peak = v;
        curves.push({ s, d });
      }

      // Four gridlines, so pick a step that divides the range cleanly and
      // keeps every tick a whole number of races.
      const yr = [0, niceStep(Math.max(peak, 4) / 4) * 4];

      axes(ctx, w, h, pad, xr, yr,
           "Race time relative to the quickest median (s)", "Races per bin");

      for (const c of curves)
        line(ctx, xs, c.d, xr, yr, pad, w, h, c.s.colour, 2.2, c.s.dash);

      // legend: colour plus dash pattern plus text, so identity survives
      // both colour blindness and a black and white print
      // It sits on an opaque panel and is sized from the measured text, so a
      // curve passing underneath can no longer run through the labels.
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "left"; ctx.textBaseline = "middle";

      const SW = 22, GAP = 8, ROW = 18, PX = 10, PY = 7;
      let tw = 0;
      for (const c of curves) tw = Math.max(tw, ctx.measureText(c.s.label).width);
      const bw = PX * 2 + SW + GAP + tw;
      const bh = PY * 2 + ROW * curves.length;
      const bx = w - pad.r - bw - 4;
      const by = pad.t + 4;

      ctx.save();
      ctx.fillStyle = PANEL; ctx.strokeStyle = GRID; ctx.lineWidth = 1;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bx, by, bw, bh, 6);
      else ctx.rect(bx, by, bw, bh);
      ctx.fill(); ctx.stroke();
      ctx.restore();

      let ly = by + PY + ROW / 2;
      for (const c of curves) {
        ctx.save();
        ctx.strokeStyle = c.s.colour; ctx.lineWidth = 2.2;
        if (c.s.dash) ctx.setLineDash(c.s.dash);
        ctx.beginPath(); ctx.moveTo(bx + PX, ly); ctx.lineTo(bx + PX + SW, ly); ctx.stroke();
        ctx.restore();
        ctx.fillStyle = MUTED; ctx.fillText(c.s.label, bx + PX + SW + GAP, ly);
        ly += ROW;
      }
    },

    /* ---------- wiring ---------- */
    init() {
      const own = ["mcPitMean", "mcPitSd", "mcSlow", "mcSc", "mcScL1", "mcVsc", "mcRf",
                   "lifeS", "lifeM", "lifeH", "mcCliff", "mcRaceSd",
                   "sdBaseS", "sdBaseM", "sdBaseH", "sdDegS", "sdDegM", "sdDegH"];
      let pending = null;
      const queue = () => { clearTimeout(pending); pending = setTimeout(() => this.render(), 200); };

      own.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        const out = document.getElementById(id + "Val");
        if (out) out.textContent = el.value;
        el.addEventListener("input", () => {
          if (out) out.textContent = el.value;
          queue();
        });
      });

      ["raceLaps", "fuelEffect", "baseS", "baseM", "baseH", "degS", "degM", "degH"]
        .forEach(id => {
          const el = document.getElementById(id);
          if (el) el.addEventListener("input", queue);
        });

      const runs = document.getElementById("mcRuns");
      if (runs) runs.addEventListener("change", () => this.render());
      const btn = document.getElementById("mcRun");
      if (btn) btn.addEventListener("click", () => this.render());
      window.addEventListener("resize", queue);

      this.render();
    }
  };

  document.addEventListener("DOMContentLoaded", () => {
    if (document.getElementById("mcChart")) mc.init();
  });
})();
