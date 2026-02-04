/* scrollytelling.js — IntersectionObserver controller + 10 SVG scene renderers */
(function () {
  'use strict';

  // ─── Helpers ────────────────────────────────────────────────
  var NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(e);
    return e;
  }

  function text(str, x, y, attrs, parent) {
    var t = el('text', Object.assign({ x: x, y: y, 'font-family': 'system-ui, sans-serif' }, attrs || {}), parent);
    t.textContent = str;
    return t;
  }

  var skipAnimations = false;

  function fadeGroup(delay) {
    var g = document.createElementNS(NS, 'g');
    if (!skipAnimations) {
      g.classList.add('svg-fade-in');
      g.style.animationDelay = delay + 's';
    }
    return g;
  }

  function sbend(x1, y1, x2, y2) {
    var my = (y1 + y2) / 2;
    return 'M' + x1 + ' ' + y1 + 'C' + x1 + ' ' + my + ' ' + x2 + ' ' + my + ' ' + x2 + ' ' + y2;
  }

  function arrow(x1, y1, x2, y2, attrs, parent) {
    var g = el('g', null, parent);
    el('line', Object.assign({
      x1: x1, y1: y1, x2: x2, y2: y2,
      stroke: 'var(--rocks-line)', 'stroke-width': 2, 'marker-end': 'url(#arrowhead)'
    }, attrs || {}), g);
    return g;
  }

  function addArrowDef(svg) {
    var defs = el('defs', null, svg);
    var marker = el('marker', {
      id: 'arrowhead', markerWidth: 10, markerHeight: 7,
      refX: 9, refY: 3.5, orient: 'auto', fill: 'var(--rocks-line)'
    }, defs);
    el('polygon', { points: '0 0, 10 3.5, 0 7' }, marker);

    // Accent arrow
    var marker2 = el('marker', {
      id: 'arrowhead-accent', markerWidth: 10, markerHeight: 7,
      refX: 9, refY: 3.5, orient: 'auto', fill: 'var(--rocks-accent)'
    }, defs);
    el('polygon', { points: '0 0, 10 3.5, 0 7' }, marker2);

    // Muted arrow
    var marker3 = el('marker', {
      id: 'arrowhead-muted', markerWidth: 10, markerHeight: 7,
      refX: 9, refY: 3.5, orient: 'auto', fill: '#B8B6AD'
    }, defs);
    el('polygon', { points: '0 0, 10 3.5, 0 7' }, marker3);
  }

  function roundRect(x, y, w, h, r, attrs, parent) {
    return el('rect', Object.assign({ x: x, y: y, width: w, height: h, rx: r, ry: r }, attrs || {}), parent);
  }

  // ─── Scene Registry ─────────────────────────────────────────
  var ScrollyScenes = {};

  // 1. Claude Terminal
  ScrollyScenes['claude-terminal'] = {
    caption: 'Coding agent loop: prompt, think, execute, observe, repeat.',
    render: function (svg) {
      addArrowDef(svg);
      // Terminal window
      var g0 = fadeGroup(0);
      svg.appendChild(g0);
      roundRect(20, 15, 460, 200, 8, { fill: '#262624', stroke: '#B8B6AD', 'stroke-width': 1.5 }, g0);
      // Title bar dots
      el('circle', { cx: 42, cy: 35, r: 6, fill: '#B8B6AD' }, g0);
      el('circle', { cx: 62, cy: 35, r: 6, fill: '#B8B6AD' }, g0);
      el('circle', { cx: 82, cy: 35, r: 6, fill: '#B8B6AD' }, g0);
      text('claude-code', 200, 38, { fill: '#B8B6AD', 'font-size': 13, 'text-anchor': 'middle', 'font-family': 'monospace' }, g0);
      // Terminal text
      var lines = ['$ /experiment "Run mup sweep"', '> Writing experiment script...', '> Executing 4 parallel jobs...', '> Analyzing results...'];
      lines.forEach(function (l, i) {
        var gl = fadeGroup(0.3 + i * 0.3);
        svg.appendChild(gl);
        text(l, 35, 72 + i * 28, { fill: i === 0 ? '#F6F6F3' : '#D5D3CB', 'font-size': 13, 'font-family': 'monospace' }, gl);
      });

      // Circular agent loop
      var cx = 250, cy = 370, r = 100;
      var labels = ['Prompt', 'Claude', 'Execute', 'Observe'];
      var angles = [-90, 0, 90, 180];
      labels.forEach(function (lbl, i) {
        var angle = angles[i] * Math.PI / 180;
        var nx = cx + r * Math.cos(angle);
        var ny = cy + r * Math.sin(angle);
        var gn = fadeGroup(0.5 + i * 0.2);
        svg.appendChild(gn);
        el('circle', { cx: nx, cy: ny, r: 28, fill: i === 1 ? 'var(--rocks-accent)' : '#F6F6F3', stroke: 'var(--rocks-line)', 'stroke-width': 1.5 }, gn);
        text(lbl, nx, ny + 5, { 'text-anchor': 'middle', fill: i === 1 ? '#fff' : 'var(--rocks-line)', 'font-size': 12, 'font-weight': 'bold' }, gn);
      });
      // Arrows between nodes
      var arrowPairs = [[0, 1], [1, 2], [2, 3], [3, 0]];
      arrowPairs.forEach(function (pair, i) {
        var a1 = angles[pair[0]] * Math.PI / 180;
        var a2 = angles[pair[1]] * Math.PI / 180;
        var x1 = cx + (r - 32) * Math.cos(a1) + 18 * Math.cos((a1 + a2) / 2);
        var y1 = cy + (r - 32) * Math.sin(a1) + 18 * Math.sin((a1 + a2) / 2);
        var x2 = cx + (r - 32) * Math.cos(a2) - 18 * Math.cos((a1 + a2) / 2);
        var y2 = cy + (r - 32) * Math.sin(a2) - 18 * Math.sin((a1 + a2) / 2);
        var ga = fadeGroup(0.7 + i * 0.15);
        svg.appendChild(ga);
        arrow(x1, y1, x2, y2, { stroke: 'var(--rocks-accent)', 'marker-end': 'url(#arrowhead-accent)', 'stroke-width': 2 }, ga);
      });
    }
  };


  // 3. Tic-tac-toe game tree
  ScrollyScenes['deductive-inductive'] = {
    caption: '',
    render: function (svg) {
      var cs = 16, bs = cs * 3; // cell size, board size

      function drawBoard(parent, bx, by, state, winCells) {
        roundRect(bx, by, bs, bs, 3, { fill: '#F6F6F3', stroke: 'var(--rocks-line)', 'stroke-width': 1 }, parent);
        for (var i = 1; i < 3; i++) {
          el('line', { x1: bx + i * cs, y1: by + 2, x2: bx + i * cs, y2: by + bs - 2, stroke: '#B8B6AD', 'stroke-width': 0.7 }, parent);
          el('line', { x1: bx + 2, y1: by + i * cs, x2: bx + bs - 2, y2: by + i * cs, stroke: '#B8B6AD', 'stroke-width': 0.7 }, parent);
        }
        var r = cs * 0.32;
        for (var c = 0; c < 9; c++) {
          if (!state[c]) continue;
          var row = Math.floor(c / 3), col = c % 3;
          var cx = bx + col * cs + cs / 2, cy = by + row * cs + cs / 2;
          if (state[c] === 1) {
            var isW = winCells && winCells.indexOf(c) >= 0;
            var xClr = isW ? 'var(--rocks-accent)' : 'var(--rocks-line)';
            var xW = isW ? 2.5 : 1.5;
            el('line', { x1: cx - r, y1: cy - r, x2: cx + r, y2: cy + r, stroke: xClr, 'stroke-width': xW, 'stroke-linecap': 'round' }, parent);
            el('line', { x1: cx + r, y1: cy - r, x2: cx - r, y2: cy + r, stroke: xClr, 'stroke-width': xW, 'stroke-linecap': 'round' }, parent);
          } else {
            el('circle', { cx: cx, cy: cy, r: r, fill: 'none', stroke: '#B8B6AD', 'stroke-width': 1.5 }, parent);
          }
        }
        if (winCells) {
          var r0 = Math.floor(winCells[0] / 3), c0 = winCells[0] % 3;
          var r2 = Math.floor(winCells[2] / 3), c2 = winCells[2] % 3;
          el('line', {
            x1: bx + c0 * cs + cs / 2, y1: by + r0 * cs + cs / 2,
            x2: bx + c2 * cs + cs / 2, y2: by + r2 * cs + cs / 2,
            stroke: 'var(--rocks-accent)', 'stroke-width': 2.5, opacity: 0.45, 'stroke-linecap': 'round'
          }, parent);
        }
      }

      function treeLine(parent, x1, y1, x2, y2, accent) {
        el('line', { x1: x1, y1: y1, x2: x2, y2: y2,
          stroke: accent ? 'var(--rocks-accent)' : '#B8B6AD',
          'stroke-width': accent ? 2 : 1, opacity: accent ? 0.8 : 0.5
        }, parent);
      }

      // Board states  (1 = X, 2 = O)
      //  0|1|2
      //  3|4|5
      //  6|7|8
      var root = [1,0,2, 0,1,0, 0,0,0]; // X center + top-right O
      // Layer 1 — O responds
      var stA  = [1,2,2, 0,1,0, 0,0,0]; // O at 1
      var stB  = [1,0,2, 2,1,0, 0,0,0]; // O at 3
      var stC  = [1,0,2, 0,1,0, 0,2,0]; // O at 7
      // Layer 2 — X responds (under A)
      var stD  = [1,2,2, 1,1,0, 0,0,0]; // X at 3
      var stE  = [1,2,2, 0,1,1, 0,0,0]; // X at 5
      var stF  = [1,2,2, 0,1,0, 0,0,1]; // X at 8 → wins diagonal 0-4-8

      // Layout coordinates
      var rootBx = 226, rootBy = 20;
      var l1Y = 135, aX = 76,  bX = 226, cX = 376;
      var l2Y = 265, dX = 5,   eX = 85,  fX = 165;

      // ── Root ──
      var g0 = fadeGroup(0);
      svg.appendChild(g0);
      drawBoard(g0, rootBx, rootBy, root);
      text('sound premise', rootBx + bs + 10, rootBy + bs / 2 + 4, { fill: 'var(--rocks-line)', 'font-size': 11, 'font-style': 'italic' }, g0);

      // ── Lines root → layer 1 ──
      var g1l = fadeGroup(0.2);
      svg.appendChild(g1l);
      treeLine(g1l, 250, rootBy + bs, aX + bs / 2, l1Y, true);
      treeLine(g1l, 250, rootBy + bs, bX + bs / 2, l1Y, false);
      treeLine(g1l, 250, rootBy + bs, cX + bs / 2, l1Y, false);
      text('deterministic,', 80, (rootBy + bs + l1Y) / 2 - 2, { fill: 'var(--rocks-accent)', 'font-size': 10, 'font-style': 'italic' }, g1l);
      text('logical rule', 80, (rootBy + bs + l1Y) / 2 + 11, { fill: 'var(--rocks-accent)', 'font-size': 10, 'font-style': 'italic' }, g1l);

      // ── Layer 1 boards ──
      var g1a = fadeGroup(0.3); svg.appendChild(g1a); drawBoard(g1a, aX, l1Y, stA);
      var g1b = fadeGroup(0.4); svg.appendChild(g1b); drawBoard(g1b, bX, l1Y, stB);
      var g1c = fadeGroup(0.5); svg.appendChild(g1c); drawBoard(g1c, cX, l1Y, stC);

      // ── Ellipsis under B, C ──
      var gd1 = fadeGroup(0.55);
      svg.appendChild(gd1);
      text('\u22EE', bX + bs / 2, l1Y + bs + 24, { 'text-anchor': 'middle', fill: '#B8B6AD', 'font-size': 18 }, gd1);
      text('\u22EE', cX + bs / 2, l1Y + bs + 24, { 'text-anchor': 'middle', fill: '#B8B6AD', 'font-size': 18 }, gd1);

      // ── Lines A → layer 2 ──
      var g2l = fadeGroup(0.6);
      svg.appendChild(g2l);
      treeLine(g2l, aX + bs / 2, l1Y + bs, dX + bs / 2, l2Y, false);
      treeLine(g2l, aX + bs / 2, l1Y + bs, eX + bs / 2, l2Y, false);
      treeLine(g2l, aX + bs / 2, l1Y + bs, fX + bs / 2, l2Y, true);

      // ── Layer 2 boards ──
      var g2d = fadeGroup(0.7); svg.appendChild(g2d); drawBoard(g2d, dX, l2Y, stD);
      var g2e = fadeGroup(0.8); svg.appendChild(g2e); drawBoard(g2e, eX, l2Y, stE);
      var g2f = fadeGroup(0.9); svg.appendChild(g2f); drawBoard(g2f, fX, l2Y, stF, [0, 4, 8]);

      // ── Ellipsis under D, E ──
      var gd2 = fadeGroup(0.95);
      svg.appendChild(gd2);
      text('\u22EE', dX + bs / 2, l2Y + bs + 24, { 'text-anchor': 'middle', fill: '#B8B6AD', 'font-size': 18 }, gd2);
      text('\u22EE', eX + bs / 2, l2Y + bs + 24, { 'text-anchor': 'middle', fill: '#B8B6AD', 'font-size': 18 }, gd2);

      // ── Win label ──
      var gW = fadeGroup(1.1);
      svg.appendChild(gW);
      text('\u2715 wins', fX + bs / 2, l2Y + bs + 22, { 'text-anchor': 'middle', fill: 'var(--rocks-accent)', 'font-size': 12, 'font-weight': 'bold' }, gW);
    }
  };

  // 3b. Inductive inference — compounding uncertainty in a belief net
  ScrollyScenes['inductive-beliefnet'] = {
    caption: '',
    render: function (svg) {
      addArrowDef(svg);

      var modules = ['Perception', 'Scene\nParsing', 'Path\nPlanning', 'Control'];
      var bw = 95, bh = 46, by = 55;
      var xs = [15, 130, 245, 360];
      var sigmas = [8, 16, 28, 44];
      var ampBase = 80;
      var baselineY = 280;

      function gaussPath(cx, sigma) {
        var amp = ampBase * sigmas[0] / sigma;
        var range = 2.5 * sigma;
        var x0 = Math.max(0, cx - range);
        var x1 = Math.min(500, cx + range);
        var n = 60;
        var d = 'M' + x0.toFixed(1) + ' ' + baselineY;
        for (var i = 0; i <= n; i++) {
          var x = x0 + (i / n) * (x1 - x0);
          var dx = x - cx;
          var y = baselineY - amp * Math.exp(-(dx * dx) / (2 * sigma * sigma));
          d += ' L' + x.toFixed(1) + ' ' + y.toFixed(1);
        }
        return d + ' L' + x1.toFixed(1) + ' ' + baselineY + ' Z';
      }

      modules.forEach(function (label, i) {
        var bx = xs[i];
        var cx = bx + bw / 2;
        var sigma = sigmas[i];

        // Box
        var gb = fadeGroup(0.1 + i * 0.15);
        svg.appendChild(gb);
        roundRect(bx, by, bw, bh, 6, { fill: '#fff', stroke: 'var(--rocks-line)', 'stroke-width': 1.5 }, gb);

        // Label (handle newline split)
        var lines = label.split('\n');
        if (lines.length === 1) {
          text(lines[0], cx, by + bh / 2 + 5, { 'text-anchor': 'middle', fill: 'var(--rocks-line)', 'font-size': 12, 'font-weight': '600' }, gb);
        } else {
          text(lines[0], cx, by + bh / 2 - 2, { 'text-anchor': 'middle', fill: 'var(--rocks-line)', 'font-size': 12, 'font-weight': '600' }, gb);
          text(lines[1], cx, by + bh / 2 + 13, { 'text-anchor': 'middle', fill: 'var(--rocks-line)', 'font-size': 12, 'font-weight': '600' }, gb);
        }

        // Arrow to next module
        if (i < modules.length - 1) {
          arrow(bx + bw + 2, by + bh / 2, xs[i + 1] - 2, by + bh / 2, {}, gb);
        }

        // Dashed line from box to gaussian
        var gd = fadeGroup(0.4 + i * 0.15);
        svg.appendChild(gd);
        el('line', {
          x1: cx, y1: by + bh, x2: cx, y2: baselineY,
          stroke: 'var(--rocks-muted)', 'stroke-width': 1, 'stroke-dasharray': '4 3'
        }, gd);

        // Gaussian curve
        var gg = fadeGroup(0.5 + i * 0.2);
        svg.appendChild(gg);
        el('path', {
          d: gaussPath(cx, sigma),
          fill: 'var(--rocks-accent-light)', stroke: 'var(--rocks-accent)', 'stroke-width': 1.5
        }, gg);
      });

      // Label
      var gl = fadeGroup(1.3);
      svg.appendChild(gl);
      text('increasing uncertainty \u2192', 250, baselineY + 28, {
        'text-anchor': 'middle', fill: '#B8B6AD', 'font-size': 11, 'font-style': 'italic'
      }, gl);
    }
  };

  // 4. AlphaGo MCTS — mini Go board tree
  ScrollyScenes['alphago-mcts'] = {
    caption: 'AlphaGo: combining search (deduction) with neural net intuition (induction).',
    render: function (svg) {
      // ── Board states (0=empty, 1=black, 2=white) ──
      var boards = {
        root: [2,0,1, 0,1,0, 2,0,0],
        A:    [2,1,1, 0,1,0, 2,0,0],
        B:    [2,0,1, 0,1,0, 2,0,1],
        C:    [2,0,1, 1,1,0, 2,0,0],
        B1:   [2,2,1, 0,1,0, 2,0,1],
        B2:   [2,0,1, 2,1,0, 2,0,1]
      };
      // Move cell for each child (index that changed from parent)
      var moves = { A: 1, B: 8, C: 3, B1: 1, B2: 3 };
      // Leaf values
      var values = { A: 0.72, C: 0.65, B1: 0.55, B2: 0.38 };
      // Policy distributions
      var policies = {
        root: [0.0, 0.30, 0.0, 0.25, 0.0, 0.10, 0.0, 0.05, 0.30],
        A:    [0.0, 0.0, 0.0, 0.35, 0.0, 0.25, 0.0, 0.20, 0.20],
        B:    [0.0, 0.35, 0.0, 0.30, 0.0, 0.15, 0.0, 0.10, 0.0],
        C:    [0.0, 0.30, 0.0, 0.0, 0.0, 0.30, 0.0, 0.20, 0.20],
        B1:   [0.0, 0.0, 0.0, 0.40, 0.0, 0.30, 0.0, 0.15, 0.0],
        B2:   [0.0, 0.25, 0.0, 0.0, 0.0, 0.35, 0.0, 0.25, 0.0]
      };

      // ── drawGoBoard ──
      var cs = 16, bs = 48, margin = 8, sr = 5;
      function drawGoBoard(parent, bx, by, state, highlight) {
        roundRect(bx, by, bs, bs, 3, { fill: '#dcb35c', stroke: '#a08030', 'stroke-width': 0.5 }, parent);
        for (var i = 0; i < 3; i++) {
          el('line', { x1: bx + margin, y1: by + margin + i * cs, x2: bx + margin + 2 * cs, y2: by + margin + i * cs, stroke: '#705020', 'stroke-width': 0.5 }, parent);
          el('line', { x1: bx + margin + i * cs, y1: by + margin, x2: bx + margin + i * cs, y2: by + margin + 2 * cs, stroke: '#705020', 'stroke-width': 0.5 }, parent);
        }
        for (var ci = 0; ci < 9; ci++) {
          if (state[ci] === 0) continue;
          var col = ci % 3, row = Math.floor(ci / 3);
          var cx = bx + margin + col * cs, cy = by + margin + row * cs;
          if (state[ci] === 1) {
            el('circle', { cx: cx, cy: cy, r: sr, fill: '#262624' }, parent);
          } else {
            el('circle', { cx: cx, cy: cy, r: sr, fill: '#F6F6F3', stroke: '#B8B6AD', 'stroke-width': 0.5 }, parent);
          }
          if (highlight === ci) {
            el('circle', { cx: cx, cy: cy, r: sr + 2.5, fill: 'none', stroke: 'var(--rocks-accent)', 'stroke-width': 1.5 }, parent);
          }
        }
      }

      // ── drawHistogram ──
      function drawHistogram(parent, hx, hy, policy) {
        var hw = 48, hh = 20, barW = hw / 9;
        var maxVal = Math.max.apply(null, policy);
        if (maxVal === 0) maxVal = 1;
        el('line', { x1: hx, y1: hy + hh, x2: hx + hw, y2: hy + hh, stroke: 'var(--rocks-muted)', 'stroke-width': 0.5 }, parent);
        for (var i = 0; i < 9; i++) {
          var barH = (policy[i] / maxVal) * hh;
          if (barH < 0.5) continue;
          el('rect', {
            x: hx + i * barW + 0.5, y: hy + hh - barH,
            width: barW - 1, height: barH,
            fill: 'var(--rocks-accent)', opacity: 0.8
          }, parent);
        }
      }

      // ── drawLeafAnnotation ──
      function drawLeafAnnotation(parent, bx, by, val, policy) {
        text('v=' + val.toFixed(2), bx + bs / 2, by + bs + 14, {
          'text-anchor': 'middle', fill: 'var(--rocks-line)',
          'font-family': 'monospace', 'font-size': 10
        }, parent);
        text('p=', bx - 2, by + bs + 35, {
          'text-anchor': 'end', fill: 'var(--rocks-muted)',
          'font-family': 'monospace', 'font-size': 9
        }, parent);
        drawHistogram(parent, bx, by + bs + 22, policy);
      }

      // ── Layout positions ──
      var pos = {
        root: { bx: 226, by: 15 },
        A:    { bx: 110, by: 135 },
        B:    { bx: 226, by: 135 },
        C:    { bx: 342, by: 135 },
        B1:   { bx: 170, by: 290 },
        B2:   { bx: 280, by: 290 }
      };

      // ── Root board (delay 0.0) ──
      var g0 = fadeGroup(0);
      svg.appendChild(g0);
      drawGoBoard(g0, pos.root.bx, pos.root.by, boards.root, -1);

      // ── Root → layer 1 edges (delay 0.2) ──
      var gEdge1 = fadeGroup(0.2);
      svg.appendChild(gEdge1);
      var rootBcx = pos.root.bx + bs / 2, rootBcy = pos.root.by + bs;
      ['A', 'B', 'C'].forEach(function (key) {
        var childCx = pos[key].bx + bs / 2, childTy = pos[key].by;
        el('path', { d: sbend(rootBcx, rootBcy, childCx, childTy), stroke: 'var(--rocks-line)', 'stroke-width': 1.5, fill: 'none' }, gEdge1);
      });

      // ── Layer 1 boards ──
      var gA = fadeGroup(0.3); svg.appendChild(gA);
      drawGoBoard(gA, pos.A.bx, pos.A.by, boards.A, moves.A);

      var gB = fadeGroup(0.4); svg.appendChild(gB);
      drawGoBoard(gB, pos.B.bx, pos.B.by, boards.B, moves.B);

      var gC = fadeGroup(0.5); svg.appendChild(gC);
      drawGoBoard(gC, pos.C.bx, pos.C.by, boards.C, moves.C);

      // ── A, C leaf annotations (delay 0.6) ──
      var gLeaf1 = fadeGroup(0.6);
      svg.appendChild(gLeaf1);
      drawLeafAnnotation(gLeaf1, pos.A.bx, pos.A.by, values.A, policies.A);
      drawLeafAnnotation(gLeaf1, pos.C.bx, pos.C.by, values.C, policies.C);

      // ── B → layer 2 edges (delay 0.7) ──
      var gEdge2 = fadeGroup(0.7);
      svg.appendChild(gEdge2);
      var bBcx = pos.B.bx + bs / 2, bBcy = pos.B.by + bs;
      ['B1', 'B2'].forEach(function (key) {
        var childCx = pos[key].bx + bs / 2, childTy = pos[key].by;
        el('path', { d: sbend(bBcx, bBcy, childCx, childTy), stroke: 'var(--rocks-line)', 'stroke-width': 1.5, fill: 'none' }, gEdge2);
      });

      // ── Layer 2 boards ──
      var gB1 = fadeGroup(0.8); svg.appendChild(gB1);
      drawGoBoard(gB1, pos.B1.bx, pos.B1.by, boards.B1, moves.B1);

      var gB2 = fadeGroup(0.9); svg.appendChild(gB2);
      drawGoBoard(gB2, pos.B2.bx, pos.B2.by, boards.B2, moves.B2);

      // ── B1, B2 leaf annotations (delay 1.0) ──
      var gLeaf2 = fadeGroup(1.0);
      svg.appendChild(gLeaf2);
      drawLeafAnnotation(gLeaf2, pos.B1.bx, pos.B1.by, values.B1, policies.B1);
      drawLeafAnnotation(gLeaf2, pos.B2.bx, pos.B2.by, values.B2, policies.B2);
    }
  };

  // 5. Reasoning Timeline
  ScrollyScenes['reasoning-timeline'] = {
    caption: 'The evolution of LLM reasoning: from prompting to RL.',
    render: function (svg) {
      addArrowDef(svg);
      // Title
      var gt = fadeGroup(0);
      svg.appendChild(gt);
      text('LLM Reasoning Timeline', 250, 40, { 'text-anchor': 'middle', fill: 'var(--rocks-line)', 'font-size': 18, 'font-weight': 'bold' }, gt);

      // Main horizontal line
      var lineG = fadeGroup(0.2);
      svg.appendChild(lineG);
      el('line', { x1: 40, y1: 250, x2: 460, y2: 250, stroke: 'var(--rocks-line)', 'stroke-width': 2.5, 'marker-end': 'url(#arrowhead)' }, lineG);

      var events = [
        { x: 70, year: '2022', label: 'Chain-of-\nThought', accent: false },
        { x: 150, year: '2023', label: 'Prompt\nHacks', accent: false },
        { x: 240, year: '2023', label: 'Process\nSupervision', accent: false },
        { x: 330, year: '2024', label: 'Tree\nSearch', accent: false },
        { x: 420, year: '2025', label: 'R1 Zero', accent: true }
      ];

      events.forEach(function (evt, i) {
        var gn = fadeGroup(0.4 + i * 0.2);
        svg.appendChild(gn);
        // Vertical tick
        el('line', { x1: evt.x, y1: 235, x2: evt.x, y2: 265, stroke: 'var(--rocks-line)', 'stroke-width': 1.5 }, gn);
        // Node
        if (evt.accent) {
          el('circle', { cx: evt.x, cy: 250, r: 12, fill: 'var(--rocks-accent)', stroke: 'var(--rocks-accent)', 'stroke-width': 2 }, gn);
          gn.querySelector('circle').classList.add('svg-pulse');
        } else {
          el('circle', { cx: evt.x, cy: 250, r: 8, fill: '#F6F6F3', stroke: 'var(--rocks-line)', 'stroke-width': 2 }, gn);
        }
        // Year below
        text(evt.year, evt.x, 290, { 'text-anchor': 'middle', fill: 'var(--rocks-muted)', 'font-size': 12 }, gn);
        // Label above
        var lines = evt.label.split('\n');
        lines.forEach(function (line, li) {
          text(line, evt.x, 210 - (lines.length - 1 - li) * 16, {
            'text-anchor': 'middle',
            fill: evt.accent ? 'var(--rocks-accent)' : 'var(--rocks-line)',
            'font-size': 13,
            'font-weight': evt.accent ? 'bold' : 'normal'
          }, gn);
        });
      });

      // Annotations
      var gaD = fadeGroup(1.6);
      svg.appendChild(gaD);
      text('"Let\'s think step by step"', 70, 330, { 'text-anchor': 'middle', fill: 'var(--rocks-muted)', 'font-size': 10, 'font-style': 'italic' }, gaD);
      text('Simple RL + rules-based rewards', 420, 330, { 'text-anchor': 'middle', fill: 'var(--rocks-accent)', 'font-size': 10, 'font-weight': 'bold' }, gaD);

      // Dead end annotation
      var gaX = fadeGroup(1.8);
      svg.appendChild(gaX);
      el('line', { x1: 120, y1: 360, x2: 190, y2: 360, stroke: '#633636', 'stroke-width': 1, 'stroke-dasharray': '4 4' }, gaX);
      text('Dead end: prospecting for lucky circuits', 250, 365, { fill: '#633636', 'font-size': 10 }, gaX);
    }
  };

  // 6. Reasoning-era: weak vs strong base LLM with RL feedback loop
  ScrollyScenes['r1-zero-recipe'] = {
    caption: '',
    animRoomMultiplier: 2.5,
    render: function (svg) {
      addArrowDef(svg);
      var cBrown = '#8B6914';
      var cGreen = '#3A7D44';
      var cMuted = '#B8B6AD';
      var cText = '#262624';
      var cBg = '#F6F6F3';
      var barH = 22, barX = 96, ansW = 38;
      var preBarW = 108;
      var genBaseW = 180;
      var fontSize = 10;

      // ── Pre-2024 (static frame) ──
      var gPre = fadeGroup(0); svg.appendChild(gPre);
      text('Pre-2024', 55, 33, { 'text-anchor': 'middle', fill: cMuted, 'font-size': 11, 'font-weight': 'bold' }, gPre);
      roundRect(12, 40, 78, 38, 4, { fill: cBg, stroke: cText, 'stroke-width': 1.5 }, gPre);
      text('Weak', 51, 55, { 'text-anchor': 'middle', fill: cText, 'font-size': 12, 'font-weight': 'bold' }, gPre);
      text('Base LLM', 51, 68, { 'text-anchor': 'middle', fill: cText, 'font-size': 12, 'font-weight': 'bold' }, gPre);
      var preGy = 48;
      // Bar track (static)
      roundRect(barX, preGy, preBarW, barH, 3, { fill: '#e8e4dc' }, gPre);

      // Dynamic pre-2024 fill bar (starts at 0 width)
      var preFill = el('rect', {
        x: barX, y: preGy, width: 0, height: barH, rx: 3, ry: 3,
        fill: cBrown, opacity: 0.7
      }, svg);
      var preBarLabel = text('less coherent thinking', barX + preBarW / 2, preGy + 14, {
        'text-anchor': 'middle', fill: '#fff', 'font-size': fontSize, opacity: 0
      }, svg);

      // Dynamic pre-2024 answer
      var preAnsX = barX + preBarW + 1;
      var preAnsGroup = el('g', { opacity: 0 }, svg);
      roundRect(preAnsX, preGy, ansW, barH, 3, { fill: cGreen }, preAnsGroup);
      text('answer', preAnsX + ansW / 2, preGy + 14, {
        'text-anchor': 'middle', fill: '#fff', 'font-size': fontSize, 'font-weight': 'bold'
      }, preAnsGroup);

      // Dynamic pre-2024 reward
      var preRwdX = preAnsX + ansW / 2;
      var preRwdGroup = el('g', { opacity: 0 }, svg);
      el('line', { x1: preRwdX, y1: preGy + barH + 14, x2: preRwdX, y2: preGy + barH + 2, stroke: cMuted, 'stroke-width': 1.5, 'marker-end': 'url(#arrowhead-muted)' }, preRwdGroup);
      text('outcome based reward', preRwdX, preGy + barH + 25, {
        'text-anchor': 'middle', fill: cMuted, 'font-size': fontSize
      }, preRwdGroup);

      // ── Post-2024 ──
      var gPost = fadeGroup(0.3); svg.appendChild(gPost);
      text('Post-2024', 55, 125, { 'text-anchor': 'middle', fill: cMuted, 'font-size': 11, 'font-weight': 'bold' }, gPost);
      var boxY = 135, boxH = 97;
      var boxBottom = boxY + boxH;
      roundRect(12, boxY, 78, boxH, 4, { fill: cBg, stroke: cText, 'stroke-width': 2 }, gPost);
      text('Strong', 51, boxY + boxH / 2 - 7, { 'text-anchor': 'middle', fill: cText, 'font-size': 12, 'font-weight': 'bold' }, gPost);
      text('Base LLM', 51, boxY + boxH / 2 + 7, { 'text-anchor': 'middle', fill: cText, 'font-size': 12, 'font-weight': 'bold' }, gPost);

      // ── Single generation row (replaces in-place each cycle) ──
      var gy = boxY + Math.round((boxH - barH) / 2);
      var genLabel = text('Gen 1', barX, gy - 5, { fill: cMuted, 'font-size': fontSize }, svg);

      // Bar track (always visible, width updates per gen)
      var barTrack = roundRect(barX, gy, genBaseW, barH, 3, { fill: '#e8e4dc' }, svg);

      // Bar fill (width animated, default to Gen 1 complete for mobile)
      var barFill = el('rect', {
        x: barX, y: gy, width: genBaseW, height: barH, rx: 3, ry: 3,
        fill: cGreen, opacity: 0.65
      }, svg);

      // Bar label
      var barLabel = text(
        'longer, more coherent thinking',
        barX + genBaseW / 2, gy + barH / 2 + 3,
        { 'text-anchor': 'middle', fill: '#fff', 'font-size': fontSize }, svg
      );

      // Answer group (default visible for mobile)
      var defaultAnsX = barX + genBaseW + 1;
      var ansGroup = el('g', null, svg);
      var ansRect = roundRect(defaultAnsX, gy, ansW, barH, 3, { fill: cGreen }, ansGroup);
      var ansText = text('answer', defaultAnsX + ansW / 2, gy + barH / 2 + 3, {
        'text-anchor': 'middle', fill: '#fff', 'font-size': fontSize, 'font-weight': 'bold'
      }, ansGroup);

      // Reward group (default visible for mobile)
      var defaultRwdX = defaultAnsX + ansW / 2;
      var rwdGroup = el('g', null, svg);
      var rwdLine = el('line', {
        x1: defaultRwdX, y1: gy + barH + 14, x2: defaultRwdX, y2: gy + barH + 2,
        stroke: cMuted, 'stroke-width': 1.5, 'marker-end': 'url(#arrowhead-muted)'
      }, rwdGroup);
      var rwdText = text('outcome based reward', defaultRwdX, gy + barH + 25, {
        'text-anchor': 'middle', fill: cMuted, 'font-size': fontSize
      }, rwdGroup);

      // RL feedback loop (hidden by default)
      var rlGroup = el('g', { opacity: 0 }, svg);
      var rlPath = el('path', {
        d: '', stroke: 'var(--rocks-accent)', 'stroke-width': 1.5, fill: 'none',
        'marker-end': 'url(#arrowhead-accent)'
      }, rlGroup);
      var rlLabel = text('RL', 170, boxBottom + 30, {
        fill: 'var(--rocks-accent)', 'font-size': 10, 'font-weight': 'bold'
      }, rlGroup);

      svg._gd = {
        preFill: preFill,
        preBarLabel: preBarLabel,
        preAnsGroup: preAnsGroup,
        preRwdGroup: preRwdGroup,
        preBarW: preBarW,
        preGy: preGy,
        genLabel: genLabel,
        barTrack: barTrack,
        barFill: barFill,
        barLabel: barLabel,
        ansGroup: ansGroup,
        ansRect: ansRect,
        ansText: ansText,
        rwdGroup: rwdGroup,
        rwdLine: rwdLine,
        rwdText: rwdText,
        rlGroup: rlGroup,
        rlPath: rlPath,
        rlLabel: rlLabel,
        gy: gy,
        boxCenterX: 51,
        boxBottomY: boxBottom
      };
    },
    scrollUpdate: function (svg, progress) {
      var d = svg._gd;
      if (!d) return;

      var barX = 96, ansW = 38, barH = 22;

      // ── Pre-2024: 3 cycles, same bar width each time (progress 0–3/7) ──
      var weakEnd = 3 / 7;
      var weakP = Math.min(1, progress / weakEnd);
      var weakCycle = Math.min(2, Math.floor(weakP * 3));
      var weakCycleP = Math.max(0, Math.min(0.999, weakP * 3 - weakCycle));

      var preFillP = Math.min(1, weakCycleP / 0.50);
      d.preFill.setAttribute('width', preFillP * d.preBarW);
      d.preBarLabel.setAttribute('opacity', preFillP > 0.35 ? '1' : '0');
      d.preAnsGroup.setAttribute('opacity', weakCycleP >= 0.55 ? '1' : '0');
      d.preRwdGroup.setAttribute('opacity', weakCycleP >= 0.68 ? '1' : '0');

      // ── Post-2024: 4 generations, each 20% longer (progress 0.30–1.0) ──
      var genWidths = [180, Math.round(180 * 1.2), Math.round(180 * 1.44), Math.round(180 * 1.728)];
      var gy = d.gy;

      var strongP = Math.max(0, Math.min(1, (progress - weakEnd) / (1 - weakEnd)));
      var genIdx = Math.min(3, Math.floor(strongP * 4));
      var cycleP = Math.max(0, Math.min(0.999, strongP * 4 - genIdx));
      var gw = genWidths[genIdx];

      // Gen label
      d.genLabel.textContent = 'Gen ' + (genIdx + 1);

      // Bar track width (snaps to current gen)
      d.barTrack.setAttribute('width', gw);

      // Bar fill: 0–55% of cycle
      var fillP = Math.min(1, cycleP / 0.55);
      d.barFill.setAttribute('width', fillP * gw);

      // Bar label centered on current bar
      d.barLabel.setAttribute('x', barX + gw / 2);
      d.barLabel.setAttribute('opacity', fillP > 0.35 ? '1' : '0');

      // Answer position + visibility
      var ansX = barX + gw + 1;
      d.ansRect.setAttribute('x', ansX);
      d.ansText.setAttribute('x', ansX + ansW / 2);
      d.ansGroup.setAttribute('opacity', cycleP >= 0.6 ? '1' : '0');

      // Reward position + visibility
      var rwdX = ansX + ansW / 2;
      d.rwdLine.setAttribute('x1', rwdX);
      d.rwdLine.setAttribute('x2', rwdX);
      d.rwdText.setAttribute('x', rwdX);
      d.rwdGroup.setAttribute('opacity', cycleP >= 0.72 ? '1' : '0');

      // RL feedback loop: U-curve from below reward to box bottom
      var rlStartY = gy + barH + 31;
      var dipY = d.boxBottomY + 33;
      d.rlPath.setAttribute('d',
        'M' + rwdX + ' ' + rlStartY +
        ' C' + rwdX + ' ' + dipY + ' ' + d.boxCenterX + ' ' + dipY +
        ' ' + d.boxCenterX + ' ' + d.boxBottomY);
      d.rlLabel.setAttribute('x', (rwdX + d.boxCenterX) / 2);
      d.rlLabel.setAttribute('y', dipY - 3);
      // Show RL loop at end of Gen 1–3, not Gen 4
      d.rlGroup.setAttribute('opacity', cycleP >= 0.88 && genIdx < 3 ? '1' : '0');
    }
  };

  // 7. Sequential Computation
  ScrollyScenes['sequential-computation'] = {
    caption: 'Sequential computation appears in forward passes, backward passes, and token generation.',
    render: function (svg) {
      addArrowDef(svg);

      var gt = fadeGroup(0);
      svg.appendChild(gt);
      text('Sequential Computation', 250, 35, { 'text-anchor': 'middle', fill: 'var(--rocks-line)', 'font-size': 16, 'font-weight': 'bold' }, gt);

      // Three columns
      var cols = [
        { x: 85, title: 'Forward\nPass', color: '#262624', direction: 'down', filled: false, dashed: false },
        { x: 250, title: 'Backward\nPass', color: '#633636', direction: 'up', filled: true, dashed: false },
        { x: 415, title: 'Token\nGeneration', color: '#B8B6AD', direction: 'right', filled: false, dashed: true }
      ];

      cols.forEach(function (col, ci) {
        var gc = fadeGroup(0.2 + ci * 0.3);
        svg.appendChild(gc);
        var titleLines = col.title.split('\n');
        titleLines.forEach(function (line, li) {
          text(line, col.x, 75 + li * 18, { 'text-anchor': 'middle', fill: col.color, 'font-size': 13, 'font-weight': 'bold' }, gc);
        });

        if (col.direction === 'down' || col.direction === 'up') {
          // Vertical stack of layers
          var layerCount = 5;
          var startY = col.direction === 'down' ? 120 : 340;
          var stepY = col.direction === 'down' ? 50 : -50;
          for (var li = 0; li < layerCount; li++) {
            var ly = startY + li * stepY;
            var gl = fadeGroup(0.5 + ci * 0.3 + li * 0.12);
            svg.appendChild(gl);
            var boxAttrs = { fill: col.filled ? col.color : '#F6F6F3', stroke: col.color, 'stroke-width': 1.5 };
            if (col.dashed) boxAttrs['stroke-dasharray'] = '4 4';
            roundRect(col.x - 40, ly - 12, 80, 28, 4, boxAttrs, gl);
            text('Layer ' + (col.direction === 'down' ? li + 1 : layerCount - li), col.x, ly + 5, { 'text-anchor': 'middle', fill: col.filled ? '#F6F6F3' : 'var(--rocks-line)', 'font-size': 11 }, gl);
            if (li < layerCount - 1) {
              var arrY1 = ly + (col.direction === 'down' ? 16 : -16);
              var arrY2 = ly + stepY + (col.direction === 'down' ? -16 : 16);
              var ga = fadeGroup(0.6 + ci * 0.3 + li * 0.12);
              svg.appendChild(ga);
              var pg = el('g', null, ga);
              el('path', { d: sbend(col.x, arrY1, col.x, arrY2), stroke: '#B8B6AD', 'stroke-width': 1.5, fill: 'none', 'marker-end': 'url(#arrowhead-muted)' }, pg);
            }
          }
        } else {
          // Horizontal token sequence
          var tokens = ['t\u2081', 't\u2082', 't\u2083', 't\u2084', 't\u2085'];
          var startX = col.x - 80;
          tokens.forEach(function (tok, ti) {
            var tx = startX + ti * 42;
            var gl = fadeGroup(0.5 + ti * 0.15);
            svg.appendChild(gl);
            var tBoxAttrs = { fill: '#F6F6F3', stroke: col.color, 'stroke-width': 1.5 };
            if (col.dashed) tBoxAttrs['stroke-dasharray'] = '4 4';
            roundRect(tx - 14, 220, 32, 28, 4, tBoxAttrs, gl);
            text(tok, tx + 2, 239, { 'text-anchor': 'middle', fill: 'var(--rocks-line)', 'font-size': 13 }, gl);
            if (ti < tokens.length - 1) {
              var ga = fadeGroup(0.6 + ti * 0.15);
              svg.appendChild(ga);
              arrow(tx + 18, 234, tx + 28, 234, { stroke: '#B8B6AD', 'stroke-width': 1.5, 'marker-end': 'url(#arrowhead-muted)' }, ga);
            }
          });
        }
      });

      // Bottom brace
      var gb = fadeGroup(1.8);
      svg.appendChild(gb);
      el('path', {
        d: 'M50,400 Q50,420 250,420 Q450,420 450,400',
        fill: 'none', stroke: 'var(--rocks-line)', 'stroke-width': 1.5
      }, gb);
      el('line', { x1: 250, y1: 420, x2: 250, y2: 435, stroke: 'var(--rocks-line)', 'stroke-width': 1.5 }, gb);
      text('"Where sequential computation runs', 250, 458, { 'text-anchor': 'middle', fill: 'var(--rocks-muted)', 'font-size': 11, 'font-style': 'italic' }, gb);
      text('along an acceptive groove"', 250, 474, { 'text-anchor': 'middle', fill: 'var(--rocks-muted)', 'font-size': 11, 'font-style': 'italic' }, gb);
    }
  };

  // 10. New Algorithms
  ScrollyScenes['new-algorithms'] = {
    caption: 'New CS primitives emerge with each era of computing.',
    render: function (svg) {
      addArrowDef(svg);

      var gt = fadeGroup(0);
      svg.appendChild(gt);
      text('CS Primitives Across Eras', 250, 35, { 'text-anchor': 'middle', fill: 'var(--rocks-line)', 'font-size': 16, 'font-weight': 'bold' }, gt);

      var eras = ['Classical', 'Deep Learning', 'Reasoning'];
      var eraColors = ['#B8B6AD', '#262624', '#633636'];
      var colWidth = 140;
      var startX = 45;

      // Column headers
      eras.forEach(function (era, i) {
        var gx = fadeGroup(0.1 + i * 0.15);
        svg.appendChild(gx);
        var cx = startX + i * (colWidth + 20) + colWidth / 2;
        text(era, cx, 75, { 'text-anchor': 'middle', fill: eraColors[i], 'font-size': 14, 'font-weight': 'bold' }, gx);
        el('line', { x1: cx - 55, y1: 85, x2: cx + 55, y2: 85, stroke: eraColors[i], 'stroke-width': 2 }, gx);
      });

      var grid = [
        ['Hash Map', 'Semantic Hash', 'Reasoning\nSearch'],
        ['Sort', 'Amortized\nSearch', 'State\nEntropy'],
        ['Monte Carlo', 'Language\nModel', 'Ask the\nLLM']
      ];

      grid.forEach(function (row, ri) {
        row.forEach(function (cell, ci) {
          var cx = startX + ci * (colWidth + 20) + colWidth / 2;
          var cy = 130 + ri * 115;
          var gi = fadeGroup(0.4 + ri * 0.2 + ci * 0.15);
          svg.appendChild(gi);

          var isReasoning = ci === 2;
          roundRect(cx - 60, cy - 25, 120, 55, 8, {
            fill: isReasoning ? 'rgba(99,54,54,0.08)' : '#F6F6F3',
            stroke: isReasoning ? '#633636' : '#B8B6AD',
            'stroke-width': isReasoning ? 2 : 1
          }, gi);

          var lines = cell.split('\n');
          lines.forEach(function (line, li) {
            text(line, cx, cy + 4 + (li - (lines.length - 1) / 2) * 16, {
              'text-anchor': 'middle',
              fill: isReasoning ? '#633636' : 'var(--rocks-line)',
              'font-size': 12,
              'font-weight': isReasoning ? 'bold' : 'normal'
            }, gi);
          });
        });
      });

      // Arrow at bottom showing progression
      var ga = fadeGroup(2.0);
      svg.appendChild(ga);
      arrow(80, 460, 420, 460, { stroke: 'var(--rocks-line)', 'stroke-width': 1.5 }, ga);
      text('Increasing abstraction & power', 250, 485, { 'text-anchor': 'middle', fill: 'var(--rocks-muted)', 'font-size': 11, 'font-style': 'italic' }, ga);
    }
  };

  // ─── Chat / Reasoning Demo Scenes ──────────────────────────

  // 5a. Pre-2022 — no reasoning, wrong answer
  ScrollyScenes['cot-evolution'] = {
    type: 'html',
    caption: 'From shooting-from-the-hip to chain-of-thought to prompt hacks.',
    render: function (svg) {
      addArrowDef(svg);
      // Phase 1 — Pre-2022
      var g1 = fadeGroup(0); svg.appendChild(g1);
      text('Pre-2022 LLM', 250, 60, { 'text-anchor': 'middle', fill: 'var(--rocks-line)', 'font-size': 14, 'font-weight': 'bold' }, g1);
      text('"What is 501 + 499 + 60?"', 250, 85, { 'text-anchor': 'middle', fill: 'var(--rocks-muted)', 'font-size': 12 }, g1);
      text('\u2192 1106  \u2717', 250, 115, { 'text-anchor': 'middle', fill: '#633636', 'font-size': 16, 'font-weight': 'bold' }, g1);
      // Phase 2 — CoT 2022
      var g2 = fadeGroup(0.3); svg.appendChild(g2);
      text('Chain-of-Thought (2022)', 250, 185, { 'text-anchor': 'middle', fill: 'var(--rocks-line)', 'font-size': 14, 'font-weight': 'bold' }, g2);
      text('"Let\'s think step by step..."', 250, 210, { 'text-anchor': 'middle', fill: '#633636', 'font-size': 12, 'font-weight': 'bold' }, g2);
      var steps = ['(501 + 499) = 1000', '(1000 + 60) = 1060', 'Answer: 1060 \u2713'];
      steps.forEach(function (s, i) {
        text(s, 250, 240 + i * 25, { 'text-anchor': 'middle', fill: i === 2 ? '#262624' : 'var(--rocks-line)', 'font-size': 12, 'font-weight': i === 2 ? 'bold' : 'normal' }, g2);
      });
      // Phase 3 — Hacks 2023
      var g3 = fadeGroup(0.6); svg.appendChild(g3);
      text('Prompt Hacks (2023)', 250, 355, { 'text-anchor': 'middle', fill: 'var(--rocks-line)', 'font-size': 14, 'font-weight': 'bold' }, g3);
      text('"A baby is going to die if you', 250, 390, { 'text-anchor': 'middle', fill: '#633636', 'font-size': 11, 'font-style': 'italic' }, g3);
      text('don\'t answer carefully"', 250, 406, { 'text-anchor': 'middle', fill: '#633636', 'font-size': 11, 'font-style': 'italic' }, g3);
      text('"I will pay you $1000 if', 250, 440, { 'text-anchor': 'middle', fill: '#B8B6AD', 'font-size': 11, 'font-style': 'italic' }, g3);
      text('you get this right"', 250, 456, { 'text-anchor': 'middle', fill: '#B8B6AD', 'font-size': 11, 'font-style': 'italic' }, g3);
    },
    renderHtml: function (container) {
      container.innerHTML = '';
      var wrapper = document.createElement('div');
      wrapper.className = 'chat-container';

      // ── Phase 1: Pre-2022 ──
      var phase1 = document.createElement('div');
      phase1.className = 'cot-phase';
      phase1.setAttribute('data-phase', '0');
      phase1.style.opacity = '0';

      var um1 = document.createElement('div');
      um1.className = 'chat-msg chat-user';
      um1.innerHTML = '<div class="chat-label">User</div><div class="chat-bubble">What is 501 + 499 + 60?</div>';
      phase1.appendChild(um1);

      var am1 = document.createElement('div');
      am1.className = 'chat-msg chat-ai';
      var label1 = document.createElement('div');
      label1.className = 'chat-label';
      label1.textContent = 'LLM';
      am1.appendChild(label1);
      var bubble1 = document.createElement('div');
      bubble1.className = 'chat-bubble';
      var tokensDiv = document.createElement('div');
      tokensDiv.className = 'chat-tokens';
      var cursor1 = document.createElement('span');
      cursor1.className = 'chat-cursor';
      cursor1.textContent = '\u258C';
      bubble1.appendChild(tokensDiv);
      bubble1.appendChild(cursor1);
      am1.appendChild(bubble1);
      var verdict1 = document.createElement('div');
      verdict1.className = 'chat-verdict';
      am1.appendChild(verdict1);
      phase1.appendChild(am1);
      wrapper.appendChild(phase1);

      // ── Phase 2: CoT 2022 ──
      var phase2 = document.createElement('div');
      phase2.className = 'cot-phase';
      phase2.setAttribute('data-phase', '1');
      phase2.style.opacity = '0';

      var um2 = document.createElement('div');
      um2.className = 'chat-msg chat-user';
      um2.innerHTML = '<div class="chat-label">User</div><div class="chat-bubble"><em>Let\'s think step by step.</em> What is 501 + 499 + 60?</div>';
      phase2.appendChild(um2);

      var am2 = document.createElement('div');
      am2.className = 'chat-msg chat-ai';
      var label2 = document.createElement('div');
      label2.className = 'chat-label';
      label2.textContent = 'LLM';
      am2.appendChild(label2);
      var bubble2 = document.createElement('div');
      bubble2.className = 'chat-bubble';
      var cotSteps = [
        '((501 + 499) + 60)',
        '(501 + 499) = 1000',
        '(1000 + 60) = 1060',
        'Answer: 1060'
      ];
      cotSteps.forEach(function (s, i) {
        var line = document.createElement('div');
        line.className = 'chat-step';
        line.textContent = s;
        line.style.opacity = '0';
        line.style.transform = 'translateY(8px)';
        line.setAttribute('data-step', i);
        bubble2.appendChild(line);
      });
      var cursor2 = document.createElement('span');
      cursor2.className = 'chat-cursor';
      cursor2.textContent = '\u258C';
      bubble2.appendChild(cursor2);
      am2.appendChild(bubble2);
      var verdict2 = document.createElement('div');
      verdict2.className = 'chat-verdict';
      am2.appendChild(verdict2);
      phase2.appendChild(am2);
      wrapper.appendChild(phase2);

      // ── Phase 3: Hacks 2023 ──
      var phase3 = document.createElement('div');
      phase3.className = 'cot-phase';
      phase3.setAttribute('data-phase', '2');
      phase3.style.opacity = '0';

      var hacks = [
        { prefix: 'A baby is going to die if you don\'t answer carefully.', color: '#633636', label: '' },
        { prefix: 'I will pay you $1000 if you get this right.', color: '#B8B6AD', label: '' }
      ];
      hacks.forEach(function (hack, i) {
        var card = document.createElement('div');
        card.className = 'chat-msg chat-user hack-card';
        card.setAttribute('data-hack', i);
        card.style.opacity = '0';
        card.style.transform = 'translateY(8px)';
        card.innerHTML = '<div class="chat-label">' + hack.label + '</div>' +
          '<div class="chat-bubble" style="border-color:' + hack.color + '">' +
          '<span style="color:' + hack.color + ';font-weight:bold">' + escapeHtml(hack.prefix) + '</span> ' +
          'What is 501 + 499 + 60?' +
          '</div>';
        phase3.appendChild(card);
      });
      wrapper.appendChild(phase3);

      container.appendChild(wrapper);

      // Stash DOM references
      container._phase1 = phase1;
      container._phase2 = phase2;
      container._phase3 = phase3;
      container._tokens = tokensDiv;
      container._cursor1 = cursor1;
      container._verdict1 = verdict1;
      container._cotSteps = bubble2.querySelectorAll('.chat-step');
      container._cursor2 = cursor2;
      container._verdict2 = verdict2;
      container._hackCards = phase3.querySelectorAll('.hack-card');
      container._lastRevealed = -1;
    },
    scrollUpdateHtml: function (container, progress) {
      var p1 = container._phase1;
      var p2 = container._phase2;
      var p3 = container._phase3;
      if (!p1) return;

      // ── Phase 1: progress 0.00–0.30 ──
      var p1Active = progress < 0.32;
      p1.style.opacity = progress >= 0.02 && p1Active ? '1' : '0';
      p1.style.display = p1Active ? '' : 'none';

      if (p1Active) {
        var chars = ['1', '1', '0', '6'];
        var tc = container._tokens;
        var revealP = Math.max(0, (progress - 0.04) / 0.18);
        var revealed = Math.floor(revealP * (chars.length + 0.5));
        revealed = Math.max(0, Math.min(chars.length, revealed));
        if (revealed !== container._lastRevealed) {
          container._lastRevealed = revealed;
          var html = '';
          for (var i = 0; i < revealed; i++) {
            html += '<span class="chat-token">' + chars[i] + '</span>';
          }
          tc.innerHTML = html;
        }
        container._cursor1.style.display = revealed > 0 && revealed < chars.length ? 'inline' : 'none';
        if (progress > 0.25) {
          container._verdict1.innerHTML = '<span class="verdict-wrong">\u2717 Incorrect \u2014 expected 1060</span>';
          container._verdict1.style.opacity = '1';
        } else {
          container._verdict1.style.opacity = '0';
        }
      }

      // ── Phase 2: progress 0.32–0.64 ──
      var p2Active = progress >= 0.32 && progress < 0.66;
      p2.style.opacity = p2Active ? '1' : '0';
      p2.style.display = progress >= 0.32 && progress < 0.66 ? '' : 'none';

      if (p2Active) {
        var steps = container._cotSteps;
        var numSteps = steps.length;
        var stepP = Math.max(0, (progress - 0.35) / 0.22);
        var revealedS = Math.floor(stepP * (numSteps + 0.5));
        revealedS = Math.max(0, Math.min(numSteps, revealedS));
        for (var si = 0; si < numSteps; si++) {
          steps[si].style.opacity = si < revealedS ? '1' : '0';
          steps[si].style.transform = si < revealedS ? 'translateY(0)' : 'translateY(8px)';
        }
        container._cursor2.style.display = revealedS > 0 && revealedS < numSteps ? 'inline' : 'none';
        if (progress > 0.60) {
          container._verdict2.innerHTML = '<span class="verdict-correct">\u2713 Correct</span>';
          container._verdict2.style.opacity = '1';
        } else {
          container._verdict2.style.opacity = '0';
        }
      }

      // ── Phase 3: progress 0.66–1.0 ──
      var p3Active = progress >= 0.66;
      p3.style.opacity = p3Active ? '1' : '0';
      p3.style.display = p3Active ? '' : 'none';

      if (p3Active) {
        var cards = container._hackCards;
        var show0 = progress > 0.70;
        if (cards[0]) {
          cards[0].style.opacity = show0 ? '1' : '0';
          cards[0].style.transform = show0 ? 'translateY(0)' : 'translateY(8px)';
        }
        var show1 = progress > 0.82;
        if (cards[1]) {
          cards[1].style.opacity = show1 ? '1' : '0';
          cards[1].style.transform = show1 ? 'translateY(0)' : 'translateY(8px)';
        }
      }
    }
  };

  // 5d. Attribution Graph — reasoning circuits (styled after Anthropic's methods-diagram)
  ScrollyScenes['attribution-graph'] = {
    caption: '',
    render: function (svg) {
      var cBg = '#B8B6AD';
      var cBadActive = '#633636';
      var cGoodActive = '#2d6a4f';
      var cNodeFill = '#F6F6F3';
      var cText = '#262624';

      // 5 token columns
      var cols = [740, 860, 980, 1100, 1220];
      var sx = function (x) { return (x - 700) * 0.79; };
      var sy = function (y) { return (y - 60) * 0.82; };
      var yOutput = 115, yL3 = 195, yL2 = 315, yL1 = 435, yEmbed = 500;

      // Title
      var gt = fadeGroup(0); svg.appendChild(gt);
      text('Prospecting for the reasoning circuit', 10, 18, { fill: cText, 'font-size': 14, 'font-weight': 'bold' }, gt);

      // Dashed separator lines
      var dashAttrs = { stroke: cBg, 'stroke-dasharray': '4 4', 'stroke-width': 1 };
      [yL1, yL2, yL3, yEmbed].forEach(function (y) {
        el('line', Object.assign({ x1: sx(710), y1: sy(y), x2: sx(1260), y2: sy(y) }, dashAttrs), svg);
      });
      el('line', Object.assign({ x1: sx(1170), y1: sy(yOutput), x2: sx(1260), y2: sy(yOutput) }, dashAttrs), svg);

      // Layer labels (right side)
      var lblX = sx(1265);
      text('\u2192 next', lblX - 8, sy(yOutput) + 4, { fill: cText, 'font-size': 9 }, svg);
      text('Layer 3', lblX, sy(yL3) + 4, { fill: cText, 'font-size': 9 }, svg);
      text('Layer 2', lblX, sy(yL2) + 4, { fill: cText, 'font-size': 9 }, svg);
      text('Layer 1', lblX, sy(yL1) + 4, { fill: cText, 'font-size': 9 }, svg);
      text('Embed', lblX, sy(yEmbed) + 4, { fill: cText, 'font-size': 9 }, svg);

      // Token labels (dynamic — stage 1: "what is 501 +499 +60")
      var tokenLabels = [];
      ['what', 'is', '501', '+499', '+60'].forEach(function (t, i) {
        tokenLabels.push(text(t, sx(cols[i]), sy(545), {
          'text-anchor': 'middle', fill: cText, 'font-size': 10, 'font-family': 'monospace'
        }, svg));
      });

      // Curved path helper
      function curvePath(x1, y1, x2, y2) {
        var my = (y1 + y2) / 2;
        return 'M' + sx(x1) + ' ' + sy(y1) +
               'C' + sx(x1) + ' ' + sy(my) + ' ' + sx(x2) + ' ' + sy(my) + ' ' + sx(x2) + ' ' + sy(y2);
      }

      // Background edges (faint network)
      var bgG = el('g', null, svg);
      [
        // embed → L1
        [740, yEmbed, 730, yL1], [740, yEmbed, 740, yL1], [740, yEmbed, 750, yL1],
        [860, yEmbed, 850, yL1], [860, yEmbed, 860, yL1], [860, yEmbed, 870, yL1],
        [980, yEmbed, 970, yL1], [980, yEmbed, 980, yL1], [980, yEmbed, 990, yL1],
        [1100, yEmbed, 1090, yL1], [1100, yEmbed, 1100, yL1], [1100, yEmbed, 1110, yL1],
        [1220, yEmbed, 1210, yL1], [1220, yEmbed, 1220, yL1], [1220, yEmbed, 1230, yL1],
        // cross-column embed → L1
        [740, yEmbed, 850, yL1], [860, yEmbed, 970, yL1], [980, yEmbed, 1090, yL1], [1100, yEmbed, 1210, yL1],
        // L1 → L2
        [730, yL1, 740, yL2], [740, yL1, 740, yL2], [750, yL1, 740, yL2],
        [850, yL1, 860, yL2], [860, yL1, 860, yL2], [870, yL1, 860, yL2],
        [970, yL1, 980, yL2], [980, yL1, 980, yL2], [990, yL1, 980, yL2],
        [1090, yL1, 1100, yL2], [1100, yL1, 1100, yL2], [1110, yL1, 1100, yL2],
        [1210, yL1, 1220, yL2], [1220, yL1, 1220, yL2], [1230, yL1, 1220, yL2],
        // cross-column L1 → L2
        [740, yL1, 860, yL2], [860, yL1, 740, yL2], [860, yL1, 980, yL2],
        [980, yL1, 1100, yL2], [1100, yL1, 1220, yL2], [1220, yL1, 1100, yL2],
        [970, yL1, 860, yL2], [1090, yL1, 860, yL2],
        // L2 → L3
        [740, yL2, 750, yL3], [740, yL2, 860, yL3],
        [860, yL2, 750, yL3], [860, yL2, 860, yL3],
        [980, yL2, 1100, yL3], [980, yL2, 860, yL3],
        [1100, yL2, 1100, yL3], [1100, yL2, 1210, yL3],
        [1220, yL2, 1100, yL3], [1220, yL2, 1210, yL3],
        [860, yL2, 750, yL3], [1100, yL2, 750, yL3],
        // L3 → output
        [750, yL3, 1200, yOutput], [860, yL3, 1200, yOutput],
        [1100, yL3, 1200, yOutput], [1210, yL3, 1200, yOutput]
      ].forEach(function (e) {
        el('path', { d: curvePath(e[0], e[1], e[2], e[3]), stroke: cBg, 'stroke-width': 0.5, opacity: 0.5, fill: 'none' }, bgG);
      });

      // Bad circuit edges (direct math-token path, cols 2-4)
      var badG = el('g', null, svg);
      [
        [980, yEmbed, 980, yL1], [1100, yEmbed, 1100, yL1], [1220, yEmbed, 1220, yL1],
        [980, yL1, 980, yL2], [1100, yL1, 1100, yL2], [1220, yL1, 1220, yL2],
        [980, yL2, 1100, yL3], [1100, yL2, 1100, yL3], [1220, yL2, 1100, yL3],
        [1100, yL3, 1200, yOutput]
      ].forEach(function (e) {
        var tier = (e[3] === yL1) ? 0 : (e[3] === yL2) ? 1 : (e[3] === yL3) ? 2 : 3;
        el('path', { d: curvePath(e[0], e[1], e[2], e[3]), stroke: cBg, 'stroke-width': 2.5, fill: 'none', 'data-tier': tier }, badG);
      });

      // Good circuit edges (CoT reasoning path — engages cols 0-1, converges through them)
      var goodG = el('g', null, svg);
      [
        [740, yEmbed, 740, yL1], [860, yEmbed, 860, yL1], [980, yEmbed, 970, yL1], [1100, yEmbed, 1090, yL1],
        [740, yL1, 740, yL2], [860, yL1, 860, yL2], [860, yL1, 740, yL2],
        [970, yL1, 860, yL2], [1090, yL1, 860, yL2],
        [740, yL2, 750, yL3], [860, yL2, 750, yL3],
        [750, yL3, 1200, yOutput]
      ].forEach(function (e) {
        var tier = (e[3] === yL1) ? 0 : (e[3] === yL2) ? 1 : (e[3] === yL3) ? 2 : 3;
        el('path', { d: curvePath(e[0], e[1], e[2], e[3]), stroke: cBg, 'stroke-width': 2.5, fill: 'none', 'data-tier': tier }, goodG);
      });

      // Nodes
      var nodeG = el('g', null, svg);
      function drawNode(x, y, bad, good, tier) {
        var filled = bad || good;
        el('circle', {
          cx: sx(x), cy: sy(y), r: filled ? 5 : 6,
          fill: filled ? cBg : cNodeFill,
          stroke: cBg, 'stroke-width': 1.5,
          'data-bad': bad ? '1' : '0',
          'data-good': good ? '1' : '0',
          'data-tier': tier
        }, nodeG);
      }

      // Embedding nodes (tier 0) — active in both circuits
      cols.forEach(function (cx) { drawNode(cx, yEmbed, true, true, 0); });
      // Layer 1 (tier 1) — 3 nodes per column
      // col 0
      drawNode(730, yL1, false, false, 1); drawNode(740, yL1, false, true, 1); drawNode(750, yL1, false, false, 1);
      // col 1
      drawNode(850, yL1, false, false, 1); drawNode(860, yL1, false, true, 1); drawNode(870, yL1, false, false, 1);
      // col 2
      drawNode(970, yL1, false, true, 1); drawNode(980, yL1, true, false, 1); drawNode(990, yL1, false, false, 1);
      // col 3
      drawNode(1090, yL1, false, true, 1); drawNode(1100, yL1, true, false, 1); drawNode(1110, yL1, false, false, 1);
      // col 4
      drawNode(1210, yL1, false, false, 1); drawNode(1220, yL1, true, false, 1); drawNode(1230, yL1, false, false, 1);
      // Layer 2 (tier 2) — one node per column
      drawNode(740, yL2, false, true, 2); drawNode(860, yL2, false, true, 2);
      drawNode(980, yL2, true, false, 2); drawNode(1100, yL2, true, false, 2); drawNode(1220, yL2, true, false, 2);
      // Layer 3 (tier 3)
      drawNode(750, yL3, false, true, 3); drawNode(860, yL3, false, false, 3);
      drawNode(1100, yL3, true, false, 3); drawNode(1210, yL3, false, false, 3);
      // Output (tier 4)
      drawNode(1200, yOutput, true, true, 4);

      // Arrowhead markers
      var defs = el('defs', null, svg);
      var badMarker = el('marker', {
        id: 'arrow-bad', viewBox: '0 0 10 10',
        refX: '10', refY: '5', markerWidth: '8', markerHeight: '8',
        orient: 'auto'
      }, defs);
      el('path', { d: 'M0,0 L10,5 L0,10 z', fill: cBadActive }, badMarker);

      var goodMarker = el('marker', {
        id: 'arrow-good', viewBox: '0 0 10 10',
        refX: '10', refY: '5', markerWidth: '8', markerHeight: '8',
        orient: 'auto'
      }, defs);
      el('path', { d: 'M0,0 L10,5 L0,10 z', fill: cGoodActive }, goodMarker);

      // Stage annotations with arrows (initially hidden)
      var badLabelG = el('g', { opacity: 0 }, svg);
      svg.appendChild(badLabelG);
      var badTxtX = 370, badTxtY = 438;
      text('low quality', badTxtX, badTxtY, {
        'text-anchor': 'middle', fill: cBadActive, 'font-size': 13, 'font-weight': 'bold', 'font-style': 'italic'
      }, badLabelG);
      text('reasoning circuit', badTxtX, badTxtY + 17, {
        'text-anchor': 'middle', fill: cBadActive, 'font-size': 13, 'font-weight': 'bold', 'font-style': 'italic'
      }, badLabelG);
      el('line', {
        x1: badTxtX, y1: badTxtY - 10,
        x2: sx(1100), y2: sy(yL2) + 16,
        stroke: cBadActive, 'stroke-width': 1.5,
        'marker-end': 'url(#arrow-bad)'
      }, badLabelG);

      var goodLabelG = el('g', { opacity: 0 }, svg);
      svg.appendChild(goodLabelG);
      var goodTxtX = 105, goodTxtY = 438;
      text('better reasoning circuit', goodTxtX, goodTxtY, {
        'text-anchor': 'middle', fill: cGoodActive, 'font-size': 13, 'font-weight': 'bold', 'font-style': 'italic'
      }, goodLabelG);
      text('(got lucky from pretraining)', goodTxtX, goodTxtY + 17, {
        'text-anchor': 'middle', fill: cGoodActive, 'font-size': 13, 'font-weight': 'bold', 'font-style': 'italic'
      }, goodLabelG);
      el('line', {
        x1: goodTxtX, y1: goodTxtY - 10,
        x2: sx(800), y2: sy(yL2) + 16,
        stroke: cGoodActive, 'stroke-width': 1.5,
        'marker-end': 'url(#arrow-good)'
      }, goodLabelG);

      // Store refs for scrollUpdate
      svg._tokenLabels = tokenLabels;
      svg._badG = badG;
      svg._goodG = goodG;
      svg._nodeG = nodeG;
      svg._badLabelG = badLabelG;
      svg._goodLabelG = goodLabelG;
    },
    scrollUpdate: function (svg, progress) {
      var cBg = '#B8B6AD';
      var cBadActive = '#633636';
      var cGoodActive = '#2d6a4f';
      var cText = '#262624';

      var tokenLabels = svg._tokenLabels;
      var badG = svg._badG;
      var goodG = svg._goodG;
      var nodeG = svg._nodeG;
      if (!tokenLabels || !badG || !goodG || !nodeG) return;

      // 6 stages across progress 0→1
      var badStart = 0.08, badEnd = 0.32;
      var badLabelAt = 0.35;
      var swapAt = 0.48;
      var goodStart = 0.52, goodEnd = 0.76;
      var goodLabelAt = 0.80;

      var isGoodPhase = progress >= swapAt;

      // Stage 1/4: update token labels
      tokenLabels[0].textContent = isGoodPhase ? 'lets' : 'what';
      tokenLabels[1].textContent = isGoodPhase ? 'think' : 'is';
      tokenLabels[0].setAttribute('fill', isGoodPhase ? cGoodActive : cText);
      tokenLabels[1].setAttribute('fill', isGoodPhase ? cGoodActive : cText);
      tokenLabels[0].setAttribute('font-weight', isGoodPhase ? 'bold' : 'normal');
      tokenLabels[1].setAttribute('font-weight', isGoodPhase ? 'bold' : 'normal');

      // Tier progress (0–5 range, each tier activates at tier + 0.8)
      var badTierP = 0;
      if (progress >= badStart && progress < swapAt) {
        badTierP = Math.min(1, (progress - badStart) / (badEnd - badStart)) * 5;
      }
      var goodTierP = 0;
      if (progress >= goodStart) {
        goodTierP = Math.min(1, (progress - goodStart) / (goodEnd - goodStart)) * 5;
      }

      // Stage 2: bad circuit edges light up
      var bEdges = badG.querySelectorAll('path');
      for (var i = 0; i < bEdges.length; i++) {
        var bTier = parseInt(bEdges[i].getAttribute('data-tier'));
        if (badTierP > bTier + 0.8) {
          bEdges[i].setAttribute('stroke', cBadActive);
          bEdges[i].setAttribute('stroke-width', '3');
        } else {
          bEdges[i].setAttribute('stroke', cBg);
          bEdges[i].setAttribute('stroke-width', '2.5');
        }
      }

      // Stage 5: good circuit edges light up
      var gEdges = goodG.querySelectorAll('path');
      for (var k = 0; k < gEdges.length; k++) {
        var gTier = parseInt(gEdges[k].getAttribute('data-tier'));
        if (goodTierP > gTier + 0.8) {
          gEdges[k].setAttribute('stroke', cGoodActive);
          gEdges[k].setAttribute('stroke-width', '3');
        } else {
          gEdges[k].setAttribute('stroke', cBg);
          gEdges[k].setAttribute('stroke-width', '2.5');
        }
      }

      // Update nodes — bad circuit first, then good circuit
      var nodes = nodeG.querySelectorAll('circle');
      for (var j = 0; j < nodes.length; j++) {
        var isBad = nodes[j].getAttribute('data-bad') === '1';
        var isGood = nodes[j].getAttribute('data-good') === '1';
        var nTier = parseInt(nodes[j].getAttribute('data-tier'));
        if (isBad && badTierP > nTier + 0.3) {
          nodes[j].setAttribute('fill', cBadActive);
          nodes[j].setAttribute('stroke', cBadActive);
        } else if (isGood && goodTierP > nTier + 0.3) {
          nodes[j].setAttribute('fill', cGoodActive);
          nodes[j].setAttribute('stroke', cGoodActive);
        } else if (isBad || isGood) {
          nodes[j].setAttribute('fill', cBg);
          nodes[j].setAttribute('stroke', cBg);
        }
      }

      // Stage 3: bad label / Stage 6: good label
      svg._badLabelG.setAttribute('opacity', (progress >= badLabelAt && progress < swapAt) ? '1' : '0');
      svg._goodLabelG.setAttribute('opacity', progress >= goodLabelAt ? '1' : '0');
    }
  };

  // ─── REPL Scenes ────────────────────────────────────────────
  var replPrompts = [
    {
      prompt: '/experiment Apply maximal update parameterization to find the best hyperparameters as I scale up. Start with GoResNet-100M as base. Use d-muP for depth-wise stability. Train 1 epoch on dev-train-100k. Submit up to 4 parallel Ray jobs.',
      thinkTime: '55m 42s',
      output: [
        'Created experiments/2025-12-27_22-18-mup-training-run.py',
        '\u2713 4 Ray jobs submitted to cluster with LR sweep',
        '\u2713 MuP parameterization with d-muP depth correction applied',
        '\u2713 Checkpoints saved every 1000 steps to R2 storage'
      ]
    },
    {
      prompt: '/experiment Run a series of experiments similar to mup-training-run.py. After each experiment, reflect on results and propose the next. Base model 10M (WIDTH=192, DEPTH=12). FLOP budget 1e15 per run. Make 10 sequential experiments and write a summary report.',
      thinkTime: '23m 37s',
      output: [
        'Completed 10/10 sequential experiments with reflection',
        '\u2713 Best config: lr=3e-4, cosine schedule, warmup=500',
        '\u2713 Policy accuracy: 34.2% \u2192 41.8% (+22% relative improvement)',
        '\u2713 Report: research_reports/2025-12-28-10m-hyperparam-tuning-summary.md'
      ]
    },
    {
      prompt: '/experiment Study scaling laws. Propose 5 model sizes from 1M to 1B params. Train each on dev-train-100k with bfloat16. Fit power law L(C) = a*C^b. Generate Chinchilla-style compute-optimal plots.',
      thinkTime: '263m 43s',
      output: [
        '5 model variants trained: 1M, 10M, 100M, 500M, 1B params',
        '\u2713 Power law fit to loss curves across FLOP budgets',
        '\u2713 Compute-optimal frontier plots generated',
        '\u2713 Scaling analysis report saved to research_reports/figures/'
      ]
    },
    {
      prompt: '/experiment Play the NN checkpoint against gnugo10. Max 81 moves per game. Evaluate 20 games. Visualize the policy softmax heatmap on each board state.',
      thinkTime: '21m 6s',
      output: [
        'nn-resnet-100m vs gnugo10: 20 games played',
        '\u2713 Win rate: 0% (0/20) against level-10 GnuGo',
        '\u2713 Average margin: -39.5 points (losses ranged 1.5 to 86)',
        '\u2713 Policy heatmaps and board visualizations saved to figures/'
      ]
    },
    {
      prompt: 'Generate 100k self-play games of gnugo10-eps vs gnugo5-eps across 4 Ray nodes with 8 workers. Save to dev-train-gnugo10-eps-100k-v2.',
      thinkTime: '4m 50s',
      output: [
        'Distributing 100,000 games across 4 nodes (25k each)',
        '\u2713 Ray job submission via ray job submit with runtime env',
        '\u2713 Sharded .npz files saved with unique game IDs',
        '\u2713 Uploaded to R2 storage after completion'
      ]
    },
    {
      prompt: 'Implement MCTS in src/alpha_go/mcts.py using AlphaGo paper nomenclature. Node with N (visit count), Q (value), logP (log-probs). Add PUCT selection, expansion, backpropagation. Test with toy environment.',
      thinkTime: '34m 24s',
      output: [
        'Created src/alpha_go/mcts.py with AlphaGo-style MCTS',
        '\u2713 Node class with N, Q, P tree data structures',
        '\u2713 PUCT selection: argmax Q(a) + c*P(a)*sqrt(N)/(1+N(a))',
        '\u2713 Tests passing for selection, expansion, and backprop in toy env'
      ]
    },
    {
      prompt: 'Speed up MCTS by implementing it in C++. Migrate Go gameplay to cpp. Use Bazel + nanobind for Python bindings. Benchmark against Python implementation and report speedups.',
      thinkTime: '56m 42s',
      output: [
        'Created src/alpha_go/cpp/ with Bazel build + nanobind bindings',
        '\u2713 C++ MCTS tree with GoGame state management',
        '\u2713 Tests verify parity with Python mcts.py implementation',
        '\u2713 Benchmark: 47x speedup for MCTS selection, 12x for full game rollout'
      ]
    },
    {
      prompt: 'Build a distributed replay buffer + data collection workers + trainer setup. Workers push episodes to buffer via gRPC. Trainer pulls i.i.d. samples. Buffer blocks if pull/push ratio exceeds 4.0. Use protobufs for serialization.',
      thinkTime: '33m 24s',
      output: [
        'Created gRPC replay buffer server with protobuf messages',
        '\u2713 InMemoryBuffer with configurable capacity and pull/push ratio',
        '\u2713 Collector workers push episodes, trainer pulls batches',
        '\u2713 Rate limiting: buffer blocks sampling above MAX_PULL_PUSH_RATIO=4.0'
      ]
    },
    {
      prompt: 'Run 20 iterations of self-play training loop: MCTS self-play (50 sims, Dirichlet noise), fine-tune 1000 steps per iteration on all accumulated data, evaluate on held-out sets. Generate heatmap of iteration vs. validation accuracy.',
      thinkTime: '8m 47s',
      output: [
        'Created experiment script for 20-iteration self-play pipeline',
        '\u2713 MCTS self-play with temperature=1.0, alpha=0.3, epsilon=0.25',
        '\u2713 Triangular heatmap: training iteration x validation datasets',
        '\u2713 Win rate tracking of each policy against prior checkpoints'
      ]
    },
    {
      prompt: "I'm building a distributed RL setup and process manager. Processes: replay buffer, offline data pusher, trainer, collectors with inference server, evaluator. Design the executor framework with WorkerStub subclasses and a FastAPI dashboard.",
      thinkTime: '242m 24s',
      output: [
        'Created src/alpha_go/distributed/ with executor framework',
        '\u2713 WorkerStub subclasses: ReplayBufferTask, InferenceServerTask, CollectorTask, TrainerTask',
        '\u2713 FastAPI dashboard showing step counters and utilization efficiency',
        '\u2713 Process lifecycle management with health checks and auto-restart',
        '\u2713 README.md documenting driver + Tasks architecture'
      ]
    },
    {
      prompt: 'Determine the critical batch size for GoResNet-500M using two methods: Gradient Noise Scale (analytical) and empirical training runs. Compute B_crit = tr(Sigma) / ||G||^2. Run pilot training with batch sizes [32, 64, 128, 256, 512].',
      thinkTime: '2m 39s',
      output: [
        'Created experiments/2025-12-26_19-45-critical-batch-size.py',
        '\u2713 Gradient Noise Scale computed via per-sample gradient variance',
        '\u2713 Empirical sweep: 5 batch sizes with loss curve comparison',
        '\u2713 B_crit estimate and diminishing returns threshold identified'
      ]
    },
    {
      prompt: "Add profiling infrastructure to infra/profiling.py. Decorator @profile('metric_name') that measures nanoseconds per call, tracks count, and saves CSV dataset on program exit for Claude to analyze bottlenecks.",
      thinkTime: '10m 6s',
      output: [
        'Created infra/profiling.py with lightweight @profile decorator',
        '\u2713 Nanosecond-precision timing with minimal overhead',
        '\u2713 Thread-safe counter accumulation across workers',
        '\u2713 Auto-saves profiling CSV on atexit for offline analysis'
      ]
    },
    {
      prompt: 'Run open-ended research: self-play data generation, validate MuP transfer from 10M to 100M, tune GoTransformer with MuP, run circuits analysis on trained models. Reflect after each experiment and propose next steps.',
      thinkTime: '1423m 20s',
      output: [
        'Ran 8+ experiments autonomously across multiple research directions',
        '\u2713 Self-play data improves policy accuracy when mixed with supervised data',
        '\u2713 MuP hyperparams transfer from 10M to 100M confirmed',
        '\u2713 GoTransformer underperforms ResNet at 10M scale - attention overhead',
        '\u2713 Reports with figures generated for each experiment'
      ]
    },
    {
      prompt: 'Train MuPGoResNet-10M on katago-vs-gnugo and train-katago-vs-katago datasets. Evaluate against gnugo and katago (10 games each). Also evaluate CppMCTSAgent with max tree depth 71. Plot results with analysis/plotting.py.',
      thinkTime: '11m 51s',
      output: [
        'Trained for 4 epochs on mixed KataGo + GnuGo game data',
        '\u2713 Validation accuracy reported every 1k steps',
        '\u2713 Eval: nn vs gnugo10 (10 games), nn vs katago (10 games)',
        '\u2713 CppMCTSAgent evaluated with depth-71 tree search'
      ]
    },
    {
      prompt: 'I think there is a bug in Go scoring. Game says W+4.5 but visually Black looks like the winner. Analyze the .npz game file and cross-reference with C++ scoring in src/alpha_go/cpp/go/go_game.h. Debug the Tromp-Taylor scoring.',
      thinkTime: '149m 46s',
      output: [
        'Loaded and visualized game board state from .npz file',
        '\u2713 Found scoring discrepancy in Tromp-Taylor territory counting',
        '\u2713 Traced bug to area scoring vs. territory scoring mismatch',
        '\u2713 Fixed go_game.h scoring implementation and verified with test cases'
      ]
    }
  ];

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  var currentReplPromptIdx = -1;
  var currentReplPhase = -1;

  // Incrementally build/tear down REPL elements based on scroll phase
  // phase 0 = prompt box only, phase 1 = + "Thinking…", phase 2 = + output & "Thought for Xm Ys"
  function setReplPhase(bodyEl, promptIdx, data, phase, isForward) {
    // Switching to a different prompt — clear everything
    if (promptIdx !== currentReplPromptIdx) {
      bodyEl.innerHTML = '';
      currentReplPromptIdx = promptIdx;
      currentReplPhase = -1;
    }

    if (phase === currentReplPhase) return;
    currentReplPhase = phase;

    var outputEl = bodyEl.querySelector('.repl-output');
    var statusEl = bodyEl.querySelector('.repl-status');
    var boxEl = bodyEl.querySelector('.repl-box');

    // Prompt box — always present
    if (!boxEl) {
      boxEl = document.createElement('div');
      boxEl.className = 'repl-box' + (isForward ? ' repl-fade-in' : '');
      boxEl.innerHTML = '<span class="repl-marker">&gt;</span><span class="repl-prompt-text">' + escapeHtml(data.prompt) + '</span>';
      bodyEl.appendChild(boxEl);
    }

    // Status — present in phase 1+
    if (phase >= 1) {
      if (!statusEl) {
        statusEl = document.createElement('div');
        statusEl.className = 'repl-status' + (isForward ? ' repl-fade-in' : '');
        bodyEl.insertBefore(statusEl, boxEl);
      }
      statusEl.textContent = phase >= 2 ? 'Thought for ' + data.thinkTime : 'Thinking\u2026';
    } else if (statusEl) {
      statusEl.remove();
    }

    // Output — present in phase 2 only
    if (phase >= 2) {
      if (!outputEl) {
        outputEl = document.createElement('div');
        outputEl.className = 'repl-output' + (isForward ? ' repl-fade-in' : '');
        data.output.forEach(function (line) {
          var cls = line.charAt(0) === '\u2713' ? 'repl-output-line success' : 'repl-output-line';
          outputEl.innerHTML += '<div class="' + cls + '">' + escapeHtml(line) + '</div>';
        });
        bodyEl.insertBefore(outputEl, statusEl || boxEl);
      }
    } else if (outputEl) {
      outputEl.remove();
    }
  }

  // ─── Stacking REPL: prompts revealed progressively within one section ───
  var replStackIndices = [0, 2, 3, 5, 6, 7, 10];
  var currentStackCount = 0;

  function renderReplStack(bodyEl, visibleCount, animate) {
    bodyEl.innerHTML = '';
    for (var i = 0; i < visibleCount; i++) {
      var data = replPrompts[replStackIndices[i]];
      var box = document.createElement('div');
      box.className = 'repl-box repl-stack-item' + (animate && i === visibleCount - 1 ? ' repl-fade-in' : '');
      box.innerHTML = '<span class="repl-marker">&gt;</span><span class="repl-prompt-text">' + escapeHtml(data.prompt) + '</span>';
      bodyEl.appendChild(box);
    }
    currentStackCount = visibleCount;
  }

  ScrollyScenes['repl-0'] = {
    type: 'repl',
    caption: '',
    promptGroup: [],
    renderRepl: function (bodyEl, animate) {
      renderReplStack(bodyEl, 1, animate !== false);
    }
  };

  // ─── Scroll Controller ──────────────────────────────────────
  function init() {
    var sections = [].slice.call(document.querySelectorAll('.scroll-section'));
    var isMobile = window.innerWidth <= 900;

    if (!sections.length) return;

    // Mobile: render static snapshots into inline containers, return early
    if (isMobile) {
      var snapshots = document.querySelectorAll('.mobile-viz-snapshot');
      snapshots.forEach(function (container) {
        var sceneName = container.getAttribute('data-scene');
        var scene = ScrollyScenes[sceneName];
        if (!scene) return;
        var mobileSvg = document.createElementNS(NS, 'svg');
        mobileSvg.setAttribute('viewBox', '0 0 500 500');
        mobileSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        container.appendChild(mobileSvg);
        addArrowDef(mobileSvg);
        scene.render(mobileSvg);
        if (scene.scrollUpdate) scene.scrollUpdate(mobileSvg, 1);
        if (scene.caption) {
          var p = document.createElement('p');
          p.style.cssText = 'font-size:0.82rem;color:#777;text-align:center;margin-top:8px;';
          p.textContent = scene.caption;
          container.appendChild(p);
        }
      });
      return;
    }

    // ── Desktop: restructure DOM for per-section viz ──

    sections.forEach(function (section) {
      // Collect all child nodes except .mobile-viz-snapshot
      var children = [];
      while (section.firstChild) {
        var child = section.firstChild;
        section.removeChild(child);
        if (child.nodeType === 1 && child.classList && child.classList.contains('mobile-viz-snapshot')) {
          // Re-append mobile snapshot at end later (hidden on desktop)
          section._mobileSnapshot = child;
        } else {
          children.push(child);
        }
      }

      // Create wrapper structure
      var inner = document.createElement('div');
      inner.className = 'scroll-section-inner';

      var textEl = document.createElement('div');
      textEl.className = 'scroll-section-text';

      var vizEl = document.createElement('div');
      vizEl.className = 'scroll-section-viz';

      // Move existing children into text container
      children.forEach(function (c) { textEl.appendChild(c); });

      // Create per-section viz elements
      var svgEl = document.createElementNS(NS, 'svg');
      svgEl.setAttribute('viewBox', '0 0 500 500');
      svgEl.setAttribute('preserveAspectRatio', 'xMidYMid meet');

      var replEl = document.createElement('div');
      replEl.className = 'scroll-viz-repl';
      var replBody = document.createElement('div');
      replBody.className = 'repl-body';
      replEl.appendChild(replBody);

      var htmlEl = document.createElement('div');
      htmlEl.className = 'scroll-viz-html';

      var captionEl = document.createElement('p');
      captionEl.className = 'scroll-viz-caption';

      vizEl.appendChild(svgEl);
      vizEl.appendChild(replEl);
      vizEl.appendChild(htmlEl);
      vizEl.appendChild(captionEl);

      inner.appendChild(textEl);
      inner.appendChild(vizEl);

      section.appendChild(inner);

      // Re-append mobile snapshot (hidden on desktop via CSS)
      if (section._mobileSnapshot) {
        section.appendChild(section._mobileSnapshot);
      }

      // Store per-section refs
      section._inner = inner;
      section._textEl = textEl;
      section._vizEl = vizEl;
      section._svgEl = svgEl;
      section._replEl = replEl;
      section._replBody = replBody;
      section._htmlEl = htmlEl;
      section._captionEl = captionEl;
      section._rendered = false;
      section._replStackCount = 0;

      // Sections with no matching scene: flow naturally, no sticky, keep empty viz for layout
      var sceneName = section.getAttribute('data-scene');
      if (!ScrollyScenes[sceneName]) {
        inner.style.position = 'static';
        inner.style.height = 'auto';
        section._noScene = true;
      }
    });

    // Measure text heights and set section heights after DOM restructure
    function measureSections() {
      var viewH = window.innerHeight;
      var triggerTop = Math.round(viewH * 0.25);
      sections.forEach(function (section) {
        // No-scene sections: natural height, no extra scroll room
        if (section._noScene) {
          section.style.height = '';
          section._textScrollMax = 0;
          section._animRoom = 0;
          return;
        }

        var textH = section._textEl.scrollHeight;
        var sceneName = section.getAttribute('data-scene');
        var scene = ScrollyScenes[sceneName];
        var hasAnim = !!(scene && (scene.scrollUpdate || scene.scrollUpdateHtml || scene.type === 'repl'));
        var textScrollMax = Math.max(0, textH - viewH);
        var animMult = (scene && scene.animRoomMultiplier) || 1;
        var animRoom = hasAnim ? Math.round(viewH * animMult) : Math.round(viewH * 0.3);
        // Animation plays over animRoom of scroll starting when section top hits 25% from viewport top.
        // Section must be tall enough for sticky viewport + whichever is larger: text scroll or anim scroll.
        var afterTrigger = Math.max(0, animRoom - triggerTop);
        section.style.height = (viewH + Math.max(textScrollMax, afterTrigger)) + 'px';
        section._textScrollMax = textScrollMax;
        section._animRoom = animRoom;
      });
    }

    measureSections();

    // Re-measure on resize
    window.addEventListener('resize', function () {
      measureSections();
    });

    // ── Activate scene into a section's viz containers ──

    function activateScene(section) {
      if (section._rendered) return;
      section._rendered = true;

      var sceneName = section.getAttribute('data-scene');
      var scene = ScrollyScenes[sceneName];
      if (!scene) return;

      if (scene.type === 'repl') {
        section._svgEl.style.display = 'none';
        section._htmlEl.style.display = 'none';
        section._replEl.style.display = 'block';
        // Don't render prompts here — scroll handler flies them in progressively
        section._replStackCount = 0;
      } else if (scene.type === 'html') {
        section._svgEl.style.display = 'none';
        section._replEl.style.display = 'none';
        section._htmlEl.style.display = 'flex';
        scene.renderHtml(section._htmlEl);
      } else {
        section._replEl.style.display = 'none';
        section._htmlEl.style.display = 'none';
        section._svgEl.style.display = '';
        while (section._svgEl.firstChild) section._svgEl.removeChild(section._svgEl.firstChild);
        addArrowDef(section._svgEl);
        scene.render(section._svgEl);
      }

      // Set caption
      section._captionEl.textContent = scene.caption || '';
    }

    // ── IntersectionObserver for lazy scene rendering ──

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          activateScene(entry.target);
        }
      });
    }, {
      rootMargin: '100px 0px 100px 0px',
      threshold: 0
    });

    sections.forEach(function (section) {
      observer.observe(section);
    });

    // ── Scroll handler: drive text scrolling + animation for ALL visible sections ──

    var rafPending = false;
    window.addEventListener('scroll', function () {
      if (!rafPending) {
        rafPending = true;
        requestAnimationFrame(function () {
          rafPending = false;
          var viewH = window.innerHeight;

          sections.forEach(function (section) {
            if (!section._rendered) return;

            var rect = section.getBoundingClientRect();

            // Skip if section not in viewport at all
            if (rect.bottom < 0 || rect.top > viewH) return;

            var scrolled = Math.max(0, -rect.top);
            var textScrollMax = section._textScrollMax;
            var animRoom = section._animRoom;
            var triggerTop = Math.round(viewH * 0.25);

            // Phase 1: text scrolling (pinned once all text visible)
            var textOffset = Math.min(scrolled, textScrollMax);
            section._textEl.style.transform = 'translateY(' + (-textOffset) + 'px)';

            // Phase 2: animation progress (begins when section top reaches 25% from viewport top)
            var animProgress = (triggerTop - rect.top) / Math.max(1, animRoom);
            animProgress = Math.max(0, Math.min(0.999, animProgress));

            // Drive scene animation
            var sceneName = section.getAttribute('data-scene');
            var scene = ScrollyScenes[sceneName];
            if (!scene) return;

            if (scene.type === 'repl') {
              var total = replStackIndices.length;
              var newCount = Math.min(total, Math.floor(animProgress * total) + 1);
              if (newCount !== section._replStackCount) {
                var isForward = newCount > section._replStackCount;
                renderReplStack(section._replBody, newCount, isForward);
                section._replStackCount = newCount;
              }
            } else if (scene.type === 'html' && scene.scrollUpdateHtml) {
              scene.scrollUpdateHtml(section._htmlEl, animProgress);
            } else if (scene.scrollUpdate) {
              scene.scrollUpdate(section._svgEl, animProgress);
            }
          });
        });
      }
    }, { passive: true });

    // Activate the first visible section immediately
    for (var i = 0; i < sections.length; i++) {
      var firstScene = sections[i].getAttribute('data-scene');
      if (firstScene && ScrollyScenes[firstScene]) {
        activateScene(sections[i]);
        break;
      }
    }
  }

  // Run on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { init(); });
  } else {
    init();
  }
})();
