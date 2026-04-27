// components/act6.jsx
// Act 6: Performance — leaf-parallel batching + game parallelism.

function Act6_Performance() {
  const { localTime: lt } = useSprite();

  // Timing phases (within ~11.5s):
  // 0..1.0    section title
  // 1.0..3.0  single tree + single leaf (baseline)
  // 3.0..5.5  leaf-parallel batching (8 leaves)
  // 5.5..7.5  "too much = diffuse" annotation
  // 7.5..11.5 game parallelism (2 trees → same server)

  // Layout: left column = one or two game trees (abstracted as a triangle w/ leaves);
  // right column = inference server (GPU).
  // Phase A: one leaf line to server.
  // Phase B: 8 leaf lines from one tree to server.
  // Phase C: 8 lines from tree A + 8 lines from tree B.

  const phaseB = clamp((lt - 1.8) / 0.8, 0, 1);  // leaf-parallel in
  const phaseC = clamp((lt - 5.0) / 0.8, 0, 1);  // diffuse annotation
  const phaseD = clamp((lt - 7.0) / 0.8, 0, 1);  // second tree in

  // Animated batch scoped to a running sim — pulse of leaves flying to server.
  const pulse = (lt % 1.2) / 1.2; // 0..1 repeating
  const pulseEase = Easing.easeInOutCubic(pulse);

  // Server position & tree positions (canvas = 1280x720, trees ~200px tall).
  // Trees pushed right of the left-column chips (which end at x ≈ 540) and
  // the "why 8, not 32?" caveat — the game label sits at cx-205, so cx must
  // be ≥ ~750 for the labels to clear those boxes.
  const serverX = 1080, serverY = 400;
  const treeAX = 780, treeAY = 220;
  const treeBX = 780, treeBY = 480;
  const TREE_LEAF_Y = 175;  // must match TREE.LEAF_Y below

  // ─── Multi-depth MCTS-like tree with distinct descent paths per leaf ───
  // Level 0: root. Level 1: 3 nodes. Level 2: ~7 nodes (varied fanout).
  // Level 3: the 8 selected leaves (one per batched sim).
  //
  // Each leaf has an explicit path through the tree so we can draw which
  // route each parallel simulation took.
  //
  // Coordinates are local to (cx, cy) and drawn as an SVG <g>.
  //
  // Tree layout (x offsets from cx, y offsets from cy):
  //   root: (0, 0)
  //   L1:   three children at (-90, 55), (0, 55), (90, 55)
  //   L2:   under L1[0] → 2 kids; under L1[1] → 3 kids; under L1[2] → 2 kids = 7
  //   L3:   8 leaves distributed: [2, 3, 3] under the L2 groups
  //
  // We hard-code paths [L1_idx, L2_idx_within_parent, leaf_idx_within_parent]
  // so 8 distinct routes light up.

  const TREE = (() => {
    // 4 leaves at different depths — illustrates depth variety.
    // Coordinates are offsets from (cx, cy).
    //
    // Tree structure (only the relevant nodes drawn):
    //   root (d=0)
    //   ├── L1a (d=1) ──── leaf #0 (d=2)            [shallow]
    //   ├── L1b (d=1)
    //   │    ├── L2b (d=2) ── leaf #1 (d=3)          [medium]
    //   │    └── L2c (d=2)
    //   │         └── L3c (d=3) ── leaf #2 (d=4)     [deep]
    //   └── L1c (d=1)
    //        └── L2d (d=2) ── leaf #3 (d=3)          [medium]
    //
    // Four paths of different lengths — exactly what we want to show.

    return {
      ROOT: { x: 0, y: 0 },
      // Descent nodes keyed by id
      NODES: {
        L1a: { x: -180, y: 50 },
        L1b: { x:  -20, y: 50 },
        L1c: { x:  180, y: 50 },
        L2b: { x:  -80, y: 105 },
        L2c: { x:   50, y: 105 },
        L2d: { x:  180, y: 105 },
        L3c: { x:   50, y: 160 },
      },
      // Leaves: list of {x,y,pathIds,depth}
      LEAVES: [
        { x: -180, y: 105, depth: 2, path: ['ROOT', 'L1a'] },
        { x:  -80, y: 160, depth: 3, path: ['ROOT', 'L1b', 'L2b'] },
        { x:   50, y: 215, depth: 4, path: ['ROOT', 'L1b', 'L2c', 'L3c'] },
        { x:  180, y: 160, depth: 3, path: ['ROOT', 'L1c', 'L2d'] },
      ],
    };
  })();

  const MiniTree = ({
    cx, cy, label,
    activeLeaves = 0,       // number of leaves (0..4) currently "being evaluated"
    leafColor = 'var(--accent-mcts)',
    color = 'var(--ink)',
    opacity = 1,
  }) => {
    const { ROOT, NODES, LEAVES } = TREE;
    const rootR = 13;
    const abs = (p) => ({ x: cx + p.x, y: cy + p.y });

    // Edges to draw as the faint skeleton: ROOT→all L1, L1→its L2s, L2→L3, etc.
    const SKELETON = [
      ['ROOT','L1a'], ['ROOT','L1b'], ['ROOT','L1c'],
      ['L1b','L2b'], ['L1b','L2c'],
      ['L1c','L2d'],
      ['L2c','L3c'],
    ];
    const nodeOf = (id) => id === 'ROOT' ? ROOT : NODES[id];

    // Map: last-segment from node → leaf
    const LEAF_SEGMENTS = LEAVES.map((lf) => {
      const last = lf.path[lf.path.length - 1];
      return { from: nodeOf(last), to: { x: lf.x, y: lf.y } };
    });

    return (
      <g opacity={opacity}>
        {/* Skeleton edges — drawn center-to-center so the highlighted
            descent paths (which are also center-to-center) overlay them
            exactly. Node shapes (root rect, circles, leaf rects) are
            rendered later and visually mask the inner part of each line. */}
        {SKELETON.map(([a, b], i) => {
          const na = abs(nodeOf(a));
          const nb = abs(nodeOf(b));
          return (
            <line key={`se${i}`}
                  x1={na.x} y1={na.y}
                  x2={nb.x} y2={nb.y}
                  stroke={color} strokeWidth={0.9} opacity={0.3} />
          );
        })}
        {/* Leaf stubs (faint) — also center-to-center. */}
        {LEAF_SEGMENTS.map((seg, i) => {
          const a = abs(seg.from), b = abs(seg.to);
          return (
            <line key={`ls${i}`}
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke={color} strokeWidth={0.7} opacity={0.25} />
          );
        })}

        {/* Highlighted descent paths for active leaves */}
        {LEAVES.map((lf, i) => {
          if (i >= activeLeaves) return null;
          const points = [...lf.path.map((id) => abs(nodeOf(id))), abs({ x: lf.x, y: lf.y })];
          const d = points.map((p, j) => `${j === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
          return (
            <path key={`hp${i}`} d={d}
                  fill="none" stroke={leafColor} strokeWidth={1.8}
                  strokeLinecap="round" strokeLinejoin="round"
                  opacity={0.85} />
          );
        })}

        {/* Root */}
        <rect x={cx - rootR} y={cy - rootR}
              width={rootR * 2} height={rootR * 2} rx={3}
              fill={color} opacity={0.88} />
        <text x={cx} y={cy + 3.5} textAnchor="middle"
              fill="var(--bg)" fontFamily="var(--mono)" fontSize={9}>root</text>

        {/* Internal nodes */}
        {Object.entries(NODES).map(([id, n]) => {
          const p = abs(n);
          return (
            <circle key={id} cx={p.x} cy={p.y} r={4.5}
                    fill="var(--bg)" stroke={color} strokeWidth={1} />
          );
        })}

        {/* Leaves */}
        {LEAVES.map((lf, i) => {
          const active = i < activeLeaves;
          const lx = cx + lf.x, ly = cy + lf.y;
          return (
            <g key={`lf${i}`}>
              <rect x={lx - 7} y={ly - 7}
                    width={14} height={14} rx={3}
                    fill={active ? leafColor : 'var(--bg)'}
                    stroke={color} strokeWidth={active ? 0 : 0.9}
                    opacity={active ? 0.95 : 0.5} />
              {/* Depth badge under each active leaf */}
              {active && (
                <text x={lx} y={ly + 26} textAnchor="middle"
                      fontFamily="var(--mono)" fontSize={9}
                      fill="var(--ink-soft)">
                  d={lf.depth}
                </text>
              )}
            </g>
          );
        })}

        {/* Game label */}
        <text x={cx - 205} y={cy - 10}
              textAnchor="start"
              fill={color} fontFamily="var(--mono)" fontSize={11}
              fontWeight={500}>{label}</text>
      </g>
    );
  };

  // Animated packets traveling from a source to the server
  const Packet = ({ x1, y1, x2, y2, t: pt, delay = 0, color = 'var(--accent-mcts)' }) => {
    const local = clamp((pt - delay) / (1 - delay), 0, 1);
    if (local <= 0 || local >= 1) return null;
    const ease = Easing.easeInOutQuad(local);
    const x = x1 + (x2 - x1) * ease;
    const y = y1 + (y2 - y1) * ease;
    return <circle cx={x} cy={y} r={3} fill={color} opacity={0.9} />;
  };

  const numLeavesFromA = phaseB < 0.3 ? 1 : 4;
  const numLeavesFromB = phaseD > 0.5 ? 4 : 0;

  return (
    <>
      <div style={{ position: 'absolute', left: 200, top: 80, opacity: clamp(lt / 0.3, 0, 1) }}>
        <SectionLabel num="09" title="Optimizing simulation throughput" />
      </div>

      <div style={{ position: 'absolute', left: 200, top: 150, maxWidth: 360, opacity: clamp((lt - 0.2) / 0.5, 0, 1) }}>
        <Caption size={13.5} style={{ lineHeight: 1.55 }}>
          MCTS is bottlenecked by neural-net evaluations. Given a fixed GPU budget, the
          goal is to maximise <em>useful</em> inferences per second — which means filling
          the GPU with as many leaves as possible <em>without</em> diffusing the search.
        </Caption>
      </div>

      {/* Two techniques, as callout chips */}
      <div style={{
        position: 'absolute', left: 200, top: 300,
        display: 'flex', flexDirection: 'column', gap: 10,
        opacity: clamp((lt - 0.8) / 0.6, 0, 1),
      }}>
        <PerfChip
          on={lt > 1.0}
          label="leaf-parallel batching"
          detail="Up to 4 leaves per batch. Each thread's descent gets a virtual-loss penalty on the nodes it visits, so the next thread picks a different path — no two threads duplicate work."
          color="var(--accent-mcts)"
        />
        <PerfChip
          on={lt > 7.0}
          label="game parallelism"
          detail="Two simultaneous games share the same inference server."
          color="var(--accent-net)"
        />
      </div>

      {/* Caveat: why not more? */}
      {lt > 5.0 && (
        <div style={{
          position: 'absolute', left: 200, bottom: 90, maxWidth: 340,
          opacity: clamp((lt - 5.0) / 0.6, 0, 1),
          padding: '10px 14px',
          border: '1px dashed var(--accent-mcts)',
          borderRadius: 8,
          background: 'rgba(255,255,255,0.3)',
        }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5,
                        color: 'var(--accent-mcts)', letterSpacing: '0.1em',
                        textTransform: 'uppercase', marginBottom: 4 }}>
            why 8, not 32?
          </div>
          <Caption size={12.5} style={{ lineHeight: 1.5 }}>
            Virtual loss lets a single search collect a batch of leaves at different
            depths, but too much parallelism makes the search diffuse — multiple
            "simultaneous" leaves duplicate work and explore worse moves.
          </Caption>
        </div>
      )}

      {/* The big diagram */}
      <svg style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} width="100%" height="100%">
        <defs>
          <marker id="perf-ar" viewBox="0 0 10 10" refX={8} refY={5}
                  markerWidth={5} markerHeight={5} orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-soft)" />
          </marker>
        </defs>

        {/* Tree A */}
        <MiniTree
          cx={treeAX} cy={treeAY}
          label="game A"
          activeLeaves={numLeavesFromA}
          leafColor="var(--accent-mcts)"
          color="var(--ink)"
          opacity={1}
        />

        {/* Tree B (game parallelism) */}
        {phaseD > 0.01 && (
          <MiniTree
            cx={treeBX} cy={treeBY}
            label="game B"
            activeLeaves={numLeavesFromB}
            leafColor="var(--accent-net)"
            color="var(--ink)"
            opacity={phaseD}
          />
        )}

        {/* Inference server box */}
        <g>
          <rect x={serverX - 110} y={serverY - 80} width={220} height={160}
                rx={12} fill="var(--bg)"
                stroke="var(--ink)" strokeWidth={1.5} />
          <text x={serverX} y={serverY - 52} textAnchor="middle"
                fill="var(--ink)" fontFamily="var(--mono)" fontSize={11}
                letterSpacing="0.1em">
            INFERENCE SERVER
          </text>
          {/* GPU glyph */}
          <NetGlyphSvg cx={serverX} cy={serverY + 5} />
          <text x={serverX} y={serverY + 52} textAnchor="middle"
                fill="var(--accent-net)" fontFamily="var(--mono)" fontSize={11}>
            f<tspan fontSize={9} dy={2}>θ</tspan><tspan dy={-2}>(s) → π, v</tspan>
          </text>
          {/* Batch indicator */}
          <text x={serverX} y={serverY + 74} textAnchor="middle"
                fill="var(--ink-soft)" fontFamily="var(--mono)" fontSize={10}>
            batch up to {phaseD > 0.5 ? '8' : '4'}
          </text>
        </g>

        {/* Leaves from tree A to server: draw connecting lines */}
        {Array.from({ length: numLeavesFromA }).map((_, i) => {
          const lf = TREE.LEAVES[i];
          const lx = treeAX + lf.x;
          const ly = treeAY + lf.y + 6;
          return (
            <line key={`la${i}`}
                  x1={lx} y1={ly}
                  x2={serverX - 110} y2={serverY - 40 + (i / 3) * 80}
                  stroke="var(--accent-mcts)" strokeWidth={0.8}
                  opacity={0.35} />
          );
        })}
        {/* Leaves from tree B to server */}
        {numLeavesFromB > 0 && Array.from({ length: numLeavesFromB }).map((_, i) => {
          const lf = TREE.LEAVES[i];
          const lx = treeBX + lf.x;
          const ly = treeBY + lf.y + 6;
          return (
            <line key={`lb${i}`}
                  x1={lx} y1={ly}
                  x2={serverX - 110} y2={serverY - 40 + (i / 3) * 80}
                  stroke="var(--accent-net)" strokeWidth={0.8}
                  opacity={0.35 * phaseD} />
          );
        })}

        {/* Animated packets — leaf → server */}
        {Array.from({ length: numLeavesFromA }).map((_, i) => {
          const lf = TREE.LEAVES[i];
          const lx = treeAX + lf.x;
          const ly = treeAY + lf.y;
          const tx = serverX - 110;
          const ty = serverY - 40 + (i / 3) * 80;
          const delay = i * 0.08;
          return (
            <Packet key={`pa${i}`}
                    x1={lx} y1={ly} x2={tx} y2={ty}
                    t={pulseEase} delay={delay}
                    color="var(--accent-mcts)" />
          );
        })}
        {numLeavesFromB > 0 && Array.from({ length: numLeavesFromB }).map((_, i) => {
          const lf = TREE.LEAVES[i];
          const lx = treeBX + lf.x;
          const ly = treeBY + lf.y;
          const tx = serverX - 110;
          const ty = serverY - 40 + (i / 3) * 80;
          const delay = i * 0.08 + 0.2;
          return (
            <Packet key={`pb${i}`}
                    x1={lx} y1={ly} x2={tx} y2={ty}
                    t={pulseEase} delay={delay}
                    color="var(--accent-net)" />
          );
        })}

        {/* Return packets from server back to trees (policy/value back) */}
        {Array.from({ length: numLeavesFromA }).map((_, i) => {
          const lf = TREE.LEAVES[i];
          const lx = treeAX + lf.x;
          const ly = treeAY + lf.y;
          const tx = serverX - 110;
          const ty = serverY - 40 + (i / 3) * 80;
          const delay = i * 0.08 + 0.55;
          return (
            <Packet key={`ra${i}`}
                    x1={tx} y1={ty} x2={lx} y2={ly}
                    t={pulseEase} delay={delay}
                    color="var(--ink-soft)" />
          );
        })}

        {/* Batch-size badge on the flow */}
        {phaseB > 0.5 && (
          <g opacity={clamp((phaseB - 0.5) / 0.5, 0, 1)}>
            <rect x={(treeAX + serverX) / 2 - 52}
                  y={treeAY + 120}
                  width={104} height={22} rx={11}
                  fill="var(--bg)"
                  stroke="var(--accent-mcts)" strokeWidth={0.8} />
            <text x={(treeAX + serverX) / 2} y={treeAY + 135}
                  textAnchor="middle"
                  fill="var(--accent-mcts)"
                  fontFamily="var(--mono)" fontSize={10.5}>
              {numLeavesFromA} leaves / batch
            </text>
          </g>
        )}
      </svg>
    </>
  );
}

function PerfChip({ on, label, detail, color }) {
  return (
    <div style={{
      opacity: on ? 1 : 0.35,
      transition: 'opacity 300ms',
      padding: '10px 14px',
      background: 'var(--bg)',
      border: `1px solid ${on ? color : 'rgba(31,26,20,0.15)'}`,
      borderLeft: `3px solid ${color}`,
      borderRadius: 6,
      maxWidth: 340,
    }}>
      <div style={{
        fontFamily: 'var(--mono)', fontSize: 11,
        color: color, fontWeight: 500,
        letterSpacing: '0.02em',
      }}>
        {label}
      </div>
      <div style={{
        fontFamily: 'var(--serif)', fontSize: 12.5,
        color: 'var(--ink-soft)',
        marginTop: 3, lineHeight: 1.4,
      }}>
        {detail}
      </div>
    </div>
  );
}

// Tiny net glyph in SVG (since NetGlyph is in div)
function NetGlyphSvg({ cx, cy }) {
  const w = 76, h = 40;
  const layers = 3, nodes = 3;
  const colW = w / (layers - 1);
  const rowH = h / (nodes - 1);
  const left = cx - w / 2, top = cy - h / 2;
  const edges = [];
  for (let c = 0; c < layers - 1; c++)
    for (let r1 = 0; r1 < nodes; r1++)
      for (let r2 = 0; r2 < nodes; r2++)
        edges.push([left + c * colW, top + r1 * rowH,
                    left + (c + 1) * colW, top + r2 * rowH]);
  return (
    <g>
      {edges.map(([x1, y1, x2, y2], i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="var(--accent-net)" strokeWidth={0.5} opacity={0.45} />
      ))}
      {Array.from({ length: layers * nodes }).map((_, i) => {
        const c = Math.floor(i / nodes), r = i % nodes;
        return (
          <circle key={i} cx={left + c * colW} cy={top + r * rowH} r={2.2}
                  fill="var(--bg)" stroke="var(--accent-net)" strokeWidth={1} />
        );
      })}
    </g>
  );
}

Object.assign(window, { Act6_Performance });
