/* Posterior explorer: interactive SBC rank histograms and corner plots.
   Reads the files written by export_web.py:
     data/manifest.json
     data/arms/<key>/summary.json
     data/arms/<key>/sims/sim<ID>.bin   (uint16 LE, n_keep x 4, scaled by lo/hi)
   Needs D3 v7 loaded before this script. */
(() => {
  "use strict";

  const DATA = "data/";
  // Okabe-Ito, same order as sbi/validation/_palette.py
  const PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"];
  // [base, subscript] pairs; rendered with <sub> in HTML and <tspan> in SVG
  const LABEL_PARTS = [["log", "10", " f", "X"], ["\u03c4"], ["r", "H/S"], ["log", "10", " M", "min"]];
  const LABELS = LABEL_PARTS.map((p) => p.map((t, i) => (i % 2 ? `<sub>${t}</sub>` : t)).join(""));
  function svgLabel(sel, j, prefix = "") {
    sel.text(prefix);
    LABEL_PARTS[j].forEach((t, i) => {
      const ts = sel.append("tspan").text(t);
      if (i % 2) ts.attr("baseline-shift", "sub").attr("font-size", "75%");
    });
    return sel;
  }
  // ticks that stay clear of panel edges, like MaxNLocator(prune="both")
  const innerTicks = (scale, n) => {
    const [a, b] = scale.domain(), pad = 0.08 * (b - a);
    return scale.ticks(n).filter((v) => v > a + pad && v < b - pad);
  };
  const LABELS_PLAIN = ["log10 fX", "tau", "rH/S", "log10 Mmin"];
  const P = 4;
  const CORNER_BINS = 30;   // _corner.py defaults
  const SMOOTH = 1.0;

  const state = {
    manifest: null,
    summaries: {},          // key -> summary.json
    color: {},              // key -> colour (stable manifest order)
    selected: [],           // keys, manifest order
    nbins: 30,
    sim: null,              // string id
    focus: null,            // key used for "list sims in this bin"
    bin: null,              // {j, b}
    sampleCache: new Map(), // "key/id" -> Float32Array[n*4]
  };

  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ data
  async function getJSON(path) {
    const r = await fetch(DATA + path);
    if (!r.ok) throw new Error(`${r.status} for ${DATA + path}`);
    return r.json();
  }

  async function getSamples(key, id) {
    const cacheKey = `${key}/${id}`;
    if (state.sampleCache.has(cacheKey)) return state.sampleCache.get(cacheKey);
    const arm = state.manifest.arms.find((a) => a.key === key);
    const r = await fetch(DATA + arm.sims.replace("{id}", id));
    if (!r.ok) throw new Error(`${r.status} for sim ${id} in ${arm.name}`);
    const q = new Uint16Array(await r.arrayBuffer());
    const s = state.summaries[key].sims[id];
    const out = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) {
      const j = i % P;
      out[i] = s.lo[j] + (q[i] / 65535) * (s.hi[j] - s.lo[j]);
    }
    state.sampleCache.set(cacheKey, out);
    return out;
  }

  function sharedSims() {
    if (!state.selected.length) return [];
    let ids = Object.keys(state.summaries[state.selected[0]].sims);
    for (const k of state.selected.slice(1)) {
      const sims = state.summaries[k].sims;
      ids = ids.filter((id) => id in sims);
    }
    return ids.sort((a, b) => a - b);
  }

  // Same binning as eval.sbc_rank
  const rankFrac = (s, j) => s.below[j] / s.n_samples;
  const rankBin = (s, j, nb) => Math.min(Math.floor(rankFrac(s, j) * nb), nb - 1);

  function rankHist(key, j, ids, nb) {
    const h = new Float64Array(nb);
    const sims = state.summaries[key].sims;
    for (const id of ids) h[rankBin(sims[id], j, nb)] += 1;
    for (let b = 0; b < nb; b++) h[b] *= nb / ids.length;
    return h;
  }

  function calibRatio(h, n) {
    const nb = h.length;
    let c = 0;
    for (const v of h) c += (v - 1) ** 2;
    return c / nb / ((nb - 1) / n);
  }

  // ------------------------------------------------------------------ hash
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    return { sim: p.get("sim"), arms: p.get("arms"), bins: p.get("bins") };
  }
  function writeHash() {
    const p = new URLSearchParams();
    if (state.sim) p.set("sim", state.sim);
    p.set("arms", state.selected.join(","));
    p.set("bins", state.nbins);
    history.replaceState(null, "", "#" + p.toString());
  }

  // ------------------------------------------------------------------ controls
  function buildControls() {
    const box = $("arm-toggles");
    box.innerHTML = "";
    state.manifest.arms.forEach((a) => {
      const label = document.createElement("label");
      label.className = "arm-toggle";
      label.innerHTML = `<input type="checkbox" value="${a.key}"> <span class="swatch" style="background:${state.color[a.key]}"></span> ${a.name} <span class="small">(${a.n_sims})</span>`;
      const cb = label.querySelector("input");
      cb.checked = state.selected.includes(a.key);
      cb.addEventListener("change", () => {
        state.selected = state.manifest.arms.map((x) => x.key)
          .filter((k) => box.querySelector(`input[value="${k}"]`).checked);
        if (!state.selected.includes(state.focus)) state.focus = state.selected[0] || null;
        const ids = sharedSims();
        if (state.sim && !ids.includes(state.sim)) state.sim = ids[0] || null;
        renderAll();
      });
      box.appendChild(label);
    });

    $("nbins").value = String(state.nbins);
    $("nbins").onchange = (e) => { state.nbins = +e.target.value; state.bin = null; renderAll(); };

    $("sim-form").onsubmit = (e) => {
      e.preventDefault();
      const v = $("sim-input").value.trim();
      if (sharedSims().includes(v)) { state.sim = v; renderAll(); setStatus(""); }
      else setStatus(`Simulation ${v} is not shared by the selected arms.`);
    };
    const step = (d) => {
      const ids = sharedSims();
      if (!ids.length) return;
      const i = Math.max(0, ids.indexOf(state.sim));
      state.sim = ids[(i + d + ids.length) % ids.length];
      renderAll();
    };
    $("sim-prev").onclick = () => step(-1);
    $("sim-next").onclick = () => step(1);
    $("sim-random").onclick = () => {
      const ids = sharedSims();
      if (ids.length) { state.sim = ids[Math.floor(Math.random() * ids.length)]; renderAll(); }
    };
    $("focus-arm").onchange = (e) => { state.focus = e.target.value; renderBinList(); };
  }

  function setStatus(msg) { $("ex-status").textContent = msg; }

  // ------------------------------------------------------------------ SBC
  function renderSBC() {
    const ids = sharedSims();
    const nb = state.nbins;
    const grid = $("sbc-grid");
    grid.innerHTML = "";
    $("sbc-count").textContent = ids.length
      ? `${ids.length} simulations shared by the selected arms, ${nb} bins.`
      : "Select at least one arm.";
    if (!ids.length) return;

    const W = 420, H = 178, m = { t: 10, r: 10, b: 36, l: 34 };
    const x = d3.scaleLinear([0, 1], [m.l, W - m.r]);
    const ymax = 2.2;
    const y = d3.scaleLinear([0, ymax], [H - m.b, m.t]);
    const s1 = 1 / Math.sqrt(ids.length / nb);

    for (let j = 0; j < P; j++) {
      const cell = document.createElement("div");
      cell.className = "sbc-cell";
      const svg = d3.select(cell).append("svg")
        .attr("viewBox", `0 0 ${W} ${H}`)
        .attr("role", "img")
        .attr("aria-label", `Rank histogram for ${LABELS_PLAIN[j]}`);

      const band = (k, op) => svg.append("rect")
        .attr("x", x(0)).attr("width", x(1) - x(0))
        .attr("y", y(Math.min(ymax, 1 + k * s1))).attr("height", y(Math.max(0, 1 - k * s1)) - y(Math.min(ymax, 1 + k * s1)))
        .attr("class", "sbc-band").attr("opacity", op);
      band(2, 0.45); band(1, 0.7);
      svg.append("line").attr("class", "sbc-unit")
        .attr("x1", x(0)).attr("x2", x(1)).attr("y1", y(1)).attr("y2", y(1));

      // highlighted bin
      if (state.bin && state.bin.j === j) {
        svg.append("rect").attr("class", "sbc-picked")
          .attr("x", x(state.bin.b / nb)).attr("width", x(1 / nb) - x(0))
          .attr("y", m.t).attr("height", H - m.b - m.t);
      }

      const chi = [];
      state.selected.forEach((key) => {
        const h = rankHist(key, j, ids, nb);
        chi.push([key, calibRatio(h, ids.length)]);
        let d = `M${x(0)},${y(Math.min(h[0], ymax))}`;
        for (let b = 0; b < nb; b++) {
          const yy = y(Math.min(h[b], ymax));
          d += `V${yy}H${x((b + 1) / nb)}`;
        }
        svg.append("path").attr("d", d).attr("class", "sbc-line").attr("stroke", state.color[key]);
      });

      // where the current simulation's truth ranks, one tick per arm
      if (state.sim) {
        state.selected.forEach((key, i) => {
          const u = rankFrac(state.summaries[key].sims[state.sim], j);
          svg.append("path")
            .attr("d", d3.symbol(d3.symbolTriangle, 38)())
            .attr("transform", `translate(${x(u)},${m.t + 6 + i * 9}) rotate(180)`)
            .attr("fill", state.color[key])
            .append("title").text(`sim ${state.sim}: rank ${u.toFixed(3)} (${state.summaries[key].name})`);
        });
      }

      // click targets
      for (let b = 0; b < nb; b++) {
        svg.append("rect").attr("class", "sbc-hit")
          .attr("x", x(b / nb)).attr("width", x(1 / nb) - x(0))
          .attr("y", m.t).attr("height", H - m.b - m.t)
          .on("click", () => { state.bin = { j, b }; renderSBC(); renderBinList(); })
          .append("title").text(`List simulations in bin ${(b / nb).toFixed(2)}–${((b + 1) / nb).toFixed(2)}`);
      }

      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`)
        .call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(y).tickValues([0, 0.5, 1, 1.5, 2]).tickSizeOuter(0));
      svg.append("text").attr("class", "axis-label").attr("x", (x(0) + x(1)) / 2).attr("y", H - 6)
        .attr("text-anchor", "middle").call(svgLabel, j, "Rank of truth in posterior, ");

      const chiRow = document.createElement("p");
      chiRow.className = "chi-row";
      chiRow.innerHTML = chi.map(([k, c]) => {
        const off = Math.abs(c - 1) > Math.sqrt(2 / (nb - 1)) ? " chi-off" : "";
        return `<span class="${off.trim()}"><span class="swatch" style="background:${state.color[k]}"></span>χ̂² ${c.toFixed(2)}</span>`;
      }).join("");
      cell.appendChild(chiRow);
      grid.appendChild(cell);
    }
  }

  function renderBinList() {
    const sel = $("focus-arm");
    sel.innerHTML = state.selected.map((k) =>
      `<option value="${k}"${k === state.focus ? " selected" : ""}>${state.summaries[k].name}</option>`).join("");
    const out = $("bin-list");
    if (!state.bin || !state.focus) {
      out.innerHTML = `<p class="small">Click a bin in any histogram to list the simulations whose truth falls in it. The edge bins hold the posteriors that miss the truth most often.</p>`;
      return;
    }
    const { j, b } = state.bin, nb = state.nbins;
    const sims = state.summaries[state.focus].sims;
    const ids = sharedSims().filter((id) => rankBin(sims[id], j, nb) === b);
    const lo = (b / nb).toFixed(2), hi = ((b + 1) / nb).toFixed(2);
    out.innerHTML = `<p class="small">${ids.length} simulations with ${LABELS[j]} rank between ${lo} and ${hi}.</p>`;
    const chips = document.createElement("div");
    chips.className = "chips";
    ids.forEach((id) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = id;
      if (id === state.sim) btn.setAttribute("aria-pressed", "true");
      btn.onclick = () => { state.sim = id; renderAll(); $("corner-title").scrollIntoView({ behavior: "smooth", block: "start" }); };
      chips.appendChild(btn);
    });
    out.appendChild(chips);
  }

  // ------------------------------------------------------------------ corner
  function percentile(sorted, p) {
    const i = (sorted.length - 1) * p;
    const lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
  }

  function gaussianBlur(H, n, sigma) {
    if (sigma <= 0) return H;
    const r = Math.ceil(4 * sigma);
    const k = [];
    let ks = 0;
    for (let i = -r; i <= r; i++) { const v = Math.exp(-0.5 * (i / sigma) ** 2); k.push(v); ks += v; }
    for (let i = 0; i < k.length; i++) k[i] /= ks;
    const reflect = (i) => { while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1; return i; };
    const tmp = new Float64Array(n * n), out = new Float64Array(n * n);
    for (let yy = 0; yy < n; yy++) for (let xx = 0; xx < n; xx++) {
      let s = 0;
      for (let t = -r; t <= r; t++) s += k[t + r] * H[yy * n + reflect(xx + t)];
      tmp[yy * n + xx] = s;
    }
    for (let yy = 0; yy < n; yy++) for (let xx = 0; xx < n; xx++) {
      let s = 0;
      for (let t = -r; t <= r; t++) s += k[t + r] * tmp[reflect(yy + t) * n + xx];
      out[yy * n + xx] = s;
    }
    return out;
  }

  // Density thresholds enclosing 95% and 68% of the mass (credible_levels in _corner.py)
  function credibleLevels(H, fracs = [0.95, 0.68]) {
    const flat = Array.from(H).sort((a, b) => b - a);
    const total = flat.reduce((a, b) => a + b, 0);
    return fracs.map((f) => {
      let c = 0;
      for (const v of flat) { c += v; if (c >= f * total) return v; }
      return flat[flat.length - 1];
    });
  }

  let cornerToken = 0;
  async function renderCorner() {
    const token = ++cornerToken;
    const holder = $("corner");
    const ids = sharedSims();
    if (!state.sim || !state.selected.length) { holder.innerHTML = ""; $("sim-table").innerHTML = ""; return; }
    $("corner-title").textContent = `Simulation ${state.sim}`;
    $("sim-input").value = state.sim;
    $("sim-pos").textContent = `${ids.indexOf(state.sim) + 1} of ${ids.length}`;

    let arms;
    try {
      arms = await Promise.all(state.selected.map(async (k) => ({ key: k, s: await getSamples(k, state.sim) })));
    } catch (err) { setStatus(err.message); return; }
    if (token !== cornerToken) return;  // a newer request superseded this one
    const truth = state.summaries[state.selected[0]].sims[state.sim].truth;

    // shared limits, as in render_corner()
    const lims = [];
    for (let j = 0; j < P; j++) {
      let lo = Infinity, hi = -Infinity;
      for (const a of arms) {
        const col = Float64Array.from({ length: a.s.length / P }, (_, i) => a.s[i * P + j]).sort();
        lo = Math.min(lo, percentile(col, 0.003));
        hi = Math.max(hi, percentile(col, 0.997));
      }
      lo = Math.min(lo, truth[j]); hi = Math.max(hi, truth[j]);
      const pad = 0.06 * (hi - lo || 1e-3);
      lims.push([lo - pad, hi + pad]);
    }

    const S = 132, G = 6, ml = 56, mb = 56, mt = 8, mr = 8;
    const W = ml + P * S + (P - 1) * G + mr, Hh = mt + P * S + (P - 1) * G + mb;
    holder.innerHTML = "";
    const svg = d3.select(holder).append("svg")
      .attr("viewBox", `0 0 ${W} ${Hh}`).attr("role", "img")
      .attr("aria-label", `Corner plot for simulation ${state.sim}`);

    const nb = CORNER_BINS;
    for (let r = 0; r < P; r++) {
      for (let c = 0; c <= r; c++) {
        const ox = ml + c * (S + G), oy = mt + r * (S + G);
        const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
        g.append("rect").attr("class", "panel").attr("width", S).attr("height", S);
        const xs = d3.scaleLinear(lims[c], [0, S]);

        if (r === c) {
          const hists = arms.map((a) => {
            const h = new Float64Array(nb), [lo, hi] = lims[c];
            for (let i = 0; i < a.s.length; i += P) {
              const b = Math.floor((a.s[i + c] - lo) / (hi - lo) * nb);
              if (b >= 0 && b < nb) h[b]++;
            }
            return h;
          });
          const ymax = d3.max(hists, (h) => d3.max(h)) * 1.1 || 1;
          const ys = d3.scaleLinear([0, ymax], [S, 0]);
          hists.forEach((h, i) => {
            let d = `M0,${S}`;
            for (let b = 0; b < nb; b++) d += `V${ys(h[b])}H${(b + 1) / nb * S}`;
            d += `V${S}`;
            g.append("path").attr("d", d).attr("class", "hist1d").attr("stroke", state.color[arms[i].key]);
          });
          g.append("line").attr("class", "truth").attr("x1", xs(truth[c])).attr("x2", xs(truth[c])).attr("y1", 0).attr("y2", S);
        } else {
          const ys = d3.scaleLinear(lims[r], [S, 0]);
          const [xlo, xhi] = lims[c], [ylo, yhi] = lims[r];
          const bw = (xhi - xlo) / nb, bh = (yhi - ylo) / nb;
          const path = d3.geoPath(d3.geoTransform({
            point(gx, gy) { this.stream.point(xs(xlo + gx * bw), ys(ylo + gy * bh)); },
          }));
          arms.forEach((a) => {
            let H = new Float64Array(nb * nb);
            for (let i = 0; i < a.s.length; i += P) {
              const bx = Math.floor((a.s[i + c] - xlo) / bw), by = Math.floor((a.s[i + r] - ylo) / bh);
              if (bx >= 0 && bx < nb && by >= 0 && by < nb) H[by * nb + bx]++;
            }
            H = gaussianBlur(H, nb, SMOOTH);
            const [l95, l68] = credibleLevels(H);
            const col = state.color[a.key];
            const contours = d3.contours().size([nb, nb]).thresholds([l95, l68])(H);
            contours.forEach((ct, k) => {
              g.append("path").attr("d", path(ct)).attr("fill", col)
                .attr("fill-opacity", k === 0 ? 0.12 : 0.3)
                .attr("stroke", col).attr("stroke-width", 1.1);
            });
          });
          g.append("line").attr("class", "truth").attr("x1", xs(truth[c])).attr("x2", xs(truth[c])).attr("y1", 0).attr("y2", S);
          g.append("line").attr("class", "truth").attr("y1", ys(truth[r])).attr("y2", ys(truth[r])).attr("x1", 0).attr("x2", S);
          g.append("path").attr("d", d3.symbol(d3.symbolStar, 42)())
            .attr("transform", `translate(${xs(truth[c])},${ys(truth[r])})`).attr("class", "truth-star");
          if (c === 0) {
            g.append("g").attr("class", "axis").call(d3.axisLeft(ys).tickValues(innerTicks(ys, 4)).tickSizeOuter(0));
            g.append("text").attr("class", "axis-label").attr("transform", `translate(${-42},${S / 2}) rotate(-90)`)
              .attr("text-anchor", "middle").call(svgLabel, r);
          }
        }
        if (r === P - 1) {
          g.append("g").attr("class", "axis").attr("transform", `translate(0,${S})`)
            .call(d3.axisBottom(xs).tickValues(innerTicks(xs, 4)).tickSizeOuter(0));
          g.append("text").attr("class", "axis-label").attr("x", S / 2).attr("y", S + 40)
            .attr("text-anchor", "middle").call(svgLabel, c);
        }
      }
    }

    // legend in the empty upper triangle
    const lg = svg.append("g").attr("transform", `translate(${ml + 2 * (S + G)},${mt + 8})`);
    arms.forEach((a, i) => {
      lg.append("rect").attr("y", i * 20).attr("width", 18).attr("height", 3).attr("fill", state.color[a.key]);
      lg.append("text").attr("class", "legend").attr("x", 26).attr("y", i * 20 + 5).text(state.summaries[a.key].name);
    });
    lg.append("line").attr("class", "truth").attr("x1", 0).attr("x2", 18).attr("y1", arms.length * 20 + 1).attr("y2", arms.length * 20 + 1);
    lg.append("text").attr("class", "legend").attr("x", 26).attr("y", arms.length * 20 + 5).text("truth");

    renderTable(truth);
  }

  function renderTable(truth) {
    const head = `<tr><th>Arm</th>${LABELS.map((l) => `<th>${l}<br><span class="small">median ± σ, rank</span></th>`).join("")}<th>GV</th></tr>`;
    const truthRow = `<tr><td>Truth</td>${truth.map((t) => `<td>${t.toFixed(3)}</td>`).join("")}<td></td></tr>`;
    const rows = state.selected.map((k) => {
      const s = state.summaries[k].sims[state.sim];
      const cells = [0, 1, 2, 3].map((j) => {
        const u = rankFrac(s, j);
        const edge = u < 0.025 || u > 0.975 ? ' class="bad"' : "";
        return `<td${edge}>${s.median[j].toFixed(3)} ± ${s.sigma[j].toFixed(3)}, ${u.toFixed(2)}</td>`;
      }).join("");
      return `<tr><td><span class="swatch" style="background:${state.color[k]}"></span>${state.summaries[k].name}</td>${cells}<td>${s.gv == null ? "–" : s.gv.toExponential(2)}</td></tr>`;
    }).join("");
    $("sim-table").innerHTML = `<table><thead>${head}</thead><tbody>${truthRow}${rows}</tbody>
      <caption>Ranks below 0.025 or above 0.975 are marked: the truth sits in the outer 5% of that posterior. Widths are the half-68% interval, in normalised coordinates.</caption></table>`;
  }

  function renderAll() {
    writeHash();
    renderSBC();
    renderBinList();
    renderCorner();
  }

  // ------------------------------------------------------------------ start
  async function init() {
    try {
      state.manifest = await getJSON("manifest.json");
    } catch (err) {
      setStatus(`Couldn't load ${DATA}manifest.json. If you opened this file directly, serve the folder instead, for example with "python -m http.server" inside docs/.`);
      return;
    }
    state.manifest.arms.forEach((a, i) => { state.color[a.key] = PALETTE[i % PALETTE.length]; });
    setStatus("Loading arm summaries…");
    const loaded = await Promise.all(state.manifest.arms.map((a) =>
      getJSON(a.summary).then((s) => [a.key, s]).catch(() => null)));
    loaded.filter(Boolean).forEach(([k, s]) => { state.summaries[k] = s; });
    state.manifest.arms = state.manifest.arms.filter((a) => a.key in state.summaries);
    state.manifest.arms.forEach((a, i) => { state.color[a.key] = PALETTE[i % PALETTE.length]; });
    if (!state.manifest.arms.length) { setStatus("No arm summaries could be loaded."); return; }
    setStatus("");

    const h = readHash();
    const keys = state.manifest.arms.map((a) => a.key);
    state.selected = h.arms ? h.arms.split(",").filter((k) => keys.includes(k)) : keys.slice();
    if (!state.selected.length) state.selected = keys.slice();
    if (h.bins && [10, 15, 20, 30, 40].includes(+h.bins)) state.nbins = +h.bins;
    state.focus = state.selected[0];
    const ids = sharedSims();
    state.sim = h.sim && ids.includes(h.sim) ? h.sim : ids[0] || null;

    buildControls();
    renderAll();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
