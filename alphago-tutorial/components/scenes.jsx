// components/scenes.jsx
// The five acts of the MCTS infographic.

// ── Shared helpers ──────────────────────────────────────────────────────────

// Example position: a 3x3 mini-board. `stones` is a fixed opening.
const ROOT_STONES = [
  { x: 0, y: 0, color: 'B' },
  { x: 2, y: 0, color: 'W' },
  { x: 0, y: 2, color: 'B' },
];

// The "true" best move is at (2,1). Prior (from weak net) and target differ.
const PRIOR = [0.18, 0.12, 0.28, 0.08, 0.06, 0.28]; // 6 bars, two candidates share weight
const VISITS = [2, 1, 6, 0, 0, 11]; // MCTS visit counts after rollouts
const TARGET = VISITS.map(v => v / VISITS.reduce((a, b) => a + b, 0));

// Smooth morph between two arrays.
function lerpArr(a, b, t) {
  return a.map((v, i) => v + (b[i] - v) * clamp(t, 0, 1));
}

// ── HeroTitle ───────────────────────────────────────────────────────────────
function HeroTitle({ showTweakPUCT }) {
  const t = useTime();
  // 0..4: visible, fade out 3..4
  const opIn = Easing.easeOutCubic(clamp(t / 0.6, 0, 1));
  const opOut = 1 - Easing.easeInCubic(clamp((t - 3.2) / 0.8, 0, 1));
  const op = Math.min(opIn, opOut);
  if (op < 0.01) return null;

  return (
    <div style={{
      position: 'absolute',
      left: 200, top: 60,
      opacity: op,
    }}>
      <div style={{
        fontFamily: 'var(--mono)',
        fontSize: 12,
        color: 'var(--ink-soft)',
        letterSpacing: '0.14em',
        textTransform: 'uppercase',
        marginBottom: 12,
      }}>
        AlphaGo · Training signal
      </div>
      <div style={{
        fontFamily: 'var(--serif)',
        fontSize: 44,
        fontWeight: 400,
        color: 'var(--ink)',
        letterSpacing: '-0.02em',
        lineHeight: 1.12,
        maxWidth: 900,
      }}>
        How MCTS builds a <em style={{ fontStyle: 'italic', color: 'var(--accent-mcts)' }}>moving target</em>
        <br/>for the policy network to chase.
      </div>
      <div style={{
        fontFamily: 'var(--serif)',
        fontSize: 17,
        color: 'var(--ink-soft)',
        maxWidth: 640,
        marginTop: 48,
        lineHeight: 1.55,
      }}>
        The network proposes a prior <span style={{ fontFamily: 'var(--mono)', color: 'var(--accent-net)' }}>p = π<sub>θ</sub>(s)</span>.
        Search refines it into sharper visit counts <span style={{ fontFamily: 'var(--mono)', color: 'var(--accent-mcts)' }}>π*</span>.
        The network trains toward π*. Repeat.
      </div>
    </div>
  );
}

// ── Act 1: Position + Prior + Value ────────────────────────────────────────
// 4..9.5s. Show position. Network evaluates. Policy bars + value gauge appear.
function Act1_PriorFromNet() {
  const t = useTime();
  const { localTime: lt } = useSprite();
  // stage layout
  const boardX = 180, boardY = 340;
  const netX = boardX + 240;
  const outX = netX + 160;

  // animation phases within 0..5.5s
  const netOn = clamp((lt - 0.6) / 0.4, 0, 1);
  const barsOn = clamp((lt - 1.4) / 0.6, 0, 1);
  const valueOn = clamp((lt - 2.0) / 0.6, 0, 1);
  const labelOn = clamp((lt - 2.6) / 0.6, 0, 1);
  const bridgeOn = clamp((lt - 4.0) / 0.7, 0, 1);

  // Build animated bar values: grow from 0
  const grown = PRIOR.map(v => v * barsOn);

  // Value v ∈ [0,1]: ease to 0.62 (agent slightly favored)
  const vTarget = 0.62;
  const vCur = vTarget * valueOn;

  return (
    <>
      {/* section label */}
      <div style={{ position: 'absolute', left: 200, top: 100, opacity: clamp(lt / 0.4, 0, 1) }}>
        <SectionLabel num="01 / Current policy" title="The network sees a position and predicts two things" />
      </div>

      {/* Board */}
      <div style={{ position: 'absolute', left: boardX, top: boardY, transform: `translate(0, ${(1 - Easing.easeOutCubic(clamp(lt / 0.5, 0, 1))) * 12}px)`, opacity: clamp(lt / 0.4, 0, 1) }}>
        <GoBoard n={3} size={160} stones={ROOT_STONES} />
        <Caption mono size={11} style={{ marginTop: 8, textAlign: 'center' }}>
          position s
        </Caption>
      </div>

      {/* Arrow: board -> net */}
      <svg style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} width="100%" height="100%">
        <ArrowCurve
          x1={boardX + 165} y1={boardY + 80}
          x2={netX - 10} y2={boardY + 80}
          arrow strokeWidth={1.3}
          color="var(--ink)"
          progress={clamp((lt - 0.4) / 0.4, 0, 1)}
          curvature={0}
        />
      </svg>

      {/* Network glyph (taller, to suggest two heads) */}
      <div style={{
        position: 'absolute', left: netX, top: boardY + 30,
        opacity: netOn,
        transform: `scale(${0.9 + 0.1 * netOn})`,
      }}>
        <div style={{
          width: 130, height: 110,
          background: 'var(--bg)',
          border: '1px dashed var(--ink-soft)',
          borderRadius: 10,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          position: 'relative',
        }}>
          <NetGlyph width={96} height={68} color="var(--accent-net)" active />
          {/* Two-head tag */}
          <div style={{
            position: 'absolute', top: 4, right: 6,
            fontFamily: 'var(--mono)', fontSize: 8.5,
            color: 'var(--ink-soft)', letterSpacing: '0.05em',
          }}>
            2 heads
          </div>
        </div>
        <Caption mono size={11} color="var(--accent-net)" style={{ marginTop: 6, textAlign: 'center', width: 130 }}>
          f<sub>θ</sub>(s)
        </Caption>
      </div>

      {/* Forking arrows: net -> policy head (top) + value head (bottom) */}
      <svg style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} width="100%" height="100%">
        <ArrowCurve
          x1={netX + 130} y1={boardY + 65}
          x2={outX - 8} y2={boardY + 38}
          arrow strokeWidth={1.3}
          color="var(--accent-net)"
          progress={clamp((lt - 1.2) / 0.4, 0, 1)}
          curvature={-0.35}
        />
        <ArrowCurve
          x1={netX + 130} y1={boardY + 95}
          x2={outX - 8} y2={boardY + 150}
          arrow strokeWidth={1.3}
          color="var(--accent-net)"
          progress={clamp((lt - 1.8) / 0.4, 0, 1)}
          curvature={0.35}
        />
      </svg>

      {/* Policy bars (prior p) — top */}
      <div style={{
        position: 'absolute', left: outX, top: boardY - 10,
        opacity: barsOn,
      }}>
        <div style={{
          padding: '10px 12px',
          background: 'var(--bg)',
          border: '1px solid var(--accent-net-soft)',
          borderRadius: 10,
        }}>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 9.5,
            color: 'var(--accent-net)', letterSpacing: '0.1em',
            textTransform: 'uppercase', marginBottom: 4,
          }}>
            policy head
          </div>
          <PolicyBars values={grown} width={130} height={46} color="var(--accent-net)" />
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 10.5,
            color: 'var(--accent-net)', marginTop: 4,
          }}>
            prior p = π<sub>θ</sub>(·|s)
          </div>
        </div>
        <Caption size={11} style={{ marginTop: 4, width: 154, color: 'var(--ink-soft)' }}>
          <em>which move should I play?</em>
        </Caption>
      </div>

      {/* Value gauge (v) — bottom */}
      <div style={{
        position: 'absolute', left: outX, top: boardY + 110,
        opacity: valueOn,
      }}>
        <div style={{
          padding: '10px 12px',
          background: 'var(--bg)',
          border: '1px solid var(--accent-net-soft)',
          borderRadius: 10,
          width: 154,
          boxSizing: 'border-box',
        }}>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 9.5,
            color: 'var(--accent-net)', letterSpacing: '0.1em',
            textTransform: 'uppercase', marginBottom: 6,
          }}>
            value head
          </div>
          {/* Horizontal probability bar: 0 = loss, 1 = win */}
          <div style={{
            position: 'relative',
            height: 18,
            background: 'var(--bg-soft)',
            borderRadius: 9,
            overflow: 'hidden',
          }}>
            <div style={{
              position: 'absolute', left: 0, top: 0, bottom: 0,
              width: `${vCur * 100}%`,
              background: 'var(--accent-net)',
              opacity: 0.85,
              transition: 'width 120ms linear',
            }} />
            <div style={{
              position: 'absolute', left: '50%', top: -2, bottom: -2,
              borderLeft: '1px dashed rgba(0,0,0,0.25)',
            }} />
          </div>
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            fontFamily: 'var(--mono)', fontSize: 9,
            color: 'var(--ink-soft)', marginTop: 3, letterSpacing: '0.05em',
          }}>
            <span>LOSS</span>
            <span style={{ color: 'var(--accent-net)', fontWeight: 600 }}>
              v = {vCur.toFixed(2)}
            </span>
            <span>WIN</span>
          </div>
        </div>
        <Caption size={11} style={{ marginTop: 4, width: 154, color: 'var(--ink-soft)' }}>
          <em>am I going to win from here?</em>
        </Caption>
      </div>

      {/* Explanatory note */}
      <div style={{
        position: 'absolute', left: 200, top: 200,
        maxWidth: 640,
        opacity: labelOn,
        transform: `translateY(${(1 - labelOn) * 8}px)`,
      }}>
        <Caption size={14.5} color="var(--ink)" style={{ lineHeight: 1.55 }}>
          One shared trunk, two heads. The <span style={{ color: 'var(--accent-net)' }}>policy head</span> outputs a distribution
          <span style={{ fontFamily: 'var(--mono)', color: 'var(--accent-net)' }}> p</span> over moves — a rough guess of what
          to play. The <span style={{ color: 'var(--accent-net)' }}>value head</span> outputs a scalar
          <span style={{ fontFamily: 'var(--mono)', color: 'var(--accent-net)' }}> v ∈ [0,1]</span> — the predicted probability
          this player wins from s. These are either initialized by training to predict human expert moves from a dataset of Go games (AlphaGo Lee) or initialized from random weights (AlphaGo Zero). Both are trained with cross-entropy loss.
        </Caption>
      </div>

    </>
  );
}

// ── Act 2: Tree search ──────────────────────────────────────────────────────
// 9..17s. Show a small tree growing: select-expand-evaluate-backup, repeated.
function Act2_TreeSearch({ showPUCT }) {
  const t = useTime();
  const { localTime: lt, duration } = useSprite();

  // Tree positions (relative to canvas). 3 levels.
  // Shifted right of the left-side caption (which extends to ~x=580) so the
  // tree has breathing room.
  const ROOT = { x: 800, y: 330 };
  const CHILDREN = [
    { x: 610, y: 480 }, // left (move A)
    { x: 800, y: 480 }, // mid  (move B) — the best
    { x: 990, y: 480 }, // right (move C)
  ];
  const GRAND = [
    { x: 720, y: 620, parent: 1 }, // under mid-left
    { x: 880, y: 620, parent: 1 }, // under mid-right
  ];

  // Simulation schedule (rollout animations). Each sim = 1.1s.
  const SIM_DUR = 1.1;
  const simCount = 6;
  const simIdx = Math.floor(lt / SIM_DUR);
  const simT = (lt % SIM_DUR) / SIM_DUR; // 0..1 inside current sim

  // Which nodes are "visited" after `k` sims (monotonically increasing)
  // We'll use a predetermined selection path per sim.
  // Each path: [root=0, childIdx+1, optional grandchildIdx+3]
  // childIdx 1=left, 2=mid, 3=right;  grandchildIdx 3=mid-left, 4=mid-right
  const PATHS = [
    [0, 2],       // root -> mid
    [0, 3],       // root -> right
    [0, 2, 3],    // root -> mid -> mid-left
    [0, 2, 4],    // root -> mid -> mid-right
    [0, 1],       // root -> left
    [0, 2, 4],    // root -> mid -> mid-right again
  ];

  // Compute accumulated visit counts
  const visitedChildren = [0, 0, 0]; // A, B, C (indices 0..2)
  const visitedGrand = [0, 0]; // mid-left, mid-right
  for (let i = 0; i < Math.min(simCount, simIdx); i++) {
    const p = PATHS[i];
    if (!p) continue;
    const c = p[1] - 1; // child index 0..2
    if (c >= 0 && c < 3) visitedChildren[c]++;
    if (p.length > 2) {
      const g = p[2] - 3;
      if (g >= 0 && g < 2) visitedGrand[g]++;
    }
  }

  // Currently-animating path
  const curPath = simIdx < simCount ? PATHS[simIdx] : PATHS[simCount - 1];
  const showRollout = simIdx < simCount;

  // Node visibility: root always, children appear after first visit,
  // grandchildren after their first visit.
  const childVisible = (i) => visitedChildren[i] > 0 || (simIdx === 0 ? false : (curPath.includes(i + 1) && simT > 0.2));
  // Simpler: a child is "born" when first selected. We'll treat presence as whether any sim <= current selects it.
  const childBorn = (i) => {
    for (let s = 0; s < simCount; s++) {
      if (s > simIdx) break;
      const p = PATHS[s];
      if (!p) continue;
      if (p[1] === i + 1) {
        if (s < simIdx) return true;
        if (s === simIdx && showRollout && simT > 0.3) return true;
      }
    }
    return false;
  };
  const grandBorn = (i) => {
    const target = i + 3;
    for (let s = 0; s < simCount; s++) {
      if (s > simIdx) break;
      const p = PATHS[s];
      if (!p) continue;
      if (p.includes(target)) {
        if (s < simIdx) return true;
        if (s === simIdx && showRollout && simT > 0.35) return true;
      }
    }
    return false;
  };

  // --- Phase labeling ---
  // simT phases: select 0.0-0.3 | expand 0.3-0.5 | evaluate 0.5-0.7 | backup 0.7-1.0
  const PHASES = [
    { key: 'select',   label: 'select',   sub: 'pick argmax Q + U',       range: [0.00, 0.30], color: 'var(--ink)' },
    { key: 'expand',   label: 'expand',   sub: 'add leaf to tree',        range: [0.30, 0.50], color: 'var(--ink)' },
    { key: 'evaluate', label: 'evaluate', sub: 'v = f_θ(s) — skip rollout', range: [0.50, 0.70], color: 'var(--accent-net)' },
    { key: 'backup',   label: 'backup',   sub: 'propagate v up path',     range: [0.70, 1.00], color: 'var(--accent-mcts)' },
  ];
  const currentPhase = showRollout
    ? PHASES.findIndex(p => simT >= p.range[0] && simT < p.range[1])
    : -1;

  // --- render ---
  return (
    <>
      <div style={{ position: 'absolute', left: 200, top: 100, opacity: clamp(lt / 0.3, 0, 1) }}>
        <SectionLabel num="03" title="Each simulation: select, expand, evaluate, back up" />
      </div>

      <div style={{ position: 'absolute', left: 200, top: 190, maxWidth: 380, opacity: clamp(lt / 0.5, 0, 1) }}>
        <Caption size={14} style={{ lineHeight: 1.55 }}>
          On each simulation, MCTS starts from the root node and walks the tree repeatedly — picking the action that
          maximises a metric <span style={{ fontFamily: 'var(--mono)' }}>PUCT(s,a) = Q(s,a) + c·p<sub>a</sub>·√N / (1+N<sub>a</sub>)</span>.
          The prior <span style={{ fontFamily: 'var(--mono)', color: 'var(--accent-net)' }}>p</span> biases exploration;
          at leaves the network's value <span style={{ fontFamily: 'var(--mono)', color: 'var(--accent-net)' }}>v</span> is
          backed up as Q. At each leaf node, the value network guesses the final outcome of the game, predicting what would happen if the tree were further expanded.
        </Caption>
        <Caption size={13} mono style={{ marginTop: 10, color: 'var(--ink-soft)' }}>
          simulation <span style={{ color: 'var(--ink)', fontWeight: 600 }}>{Math.min(simIdx + (showRollout ? 1 : 0), simCount)}</span> / {simCount}
        </Caption>
      </div>

      {/* Phase stepper — shows current step of the 4-step MCTS simulation */}
      {showRollout && currentPhase >= 0 && (
        <div style={{
          position: 'absolute', left: 200, top: 400, width: 380,
          opacity: clamp((lt - 0.3) / 0.4, 0, 1),
        }}>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 10.5,
            letterSpacing: '0.14em', textTransform: 'uppercase',
            color: 'var(--ink-soft)', marginBottom: 10,
          }}>
            one simulation = four steps
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {PHASES.map((p, i) => {
              const active = i === currentPhase;
              const done = i < currentPhase;
              const localProg = active
                ? clamp((simT - p.range[0]) / (p.range[1] - p.range[0]), 0, 1)
                : done ? 1 : 0;
              return (
                <div key={p.key} style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '6px 10px',
                  background: active ? 'rgba(31,26,20,0.05)' : 'transparent',
                  borderLeft: `2px solid ${active ? p.color : (done ? 'var(--ink-soft)' : 'rgba(31,26,20,0.1)')}`,
                  borderRadius: '0 4px 4px 0',
                  transition: 'background 180ms',
                }}>
                  <div style={{
                    fontFamily: 'var(--mono)', fontSize: 10,
                    color: active ? p.color : (done ? 'var(--ink-soft)' : 'rgba(31,26,20,0.35)'),
                    fontWeight: 600,
                    minWidth: 14,
                  }}>
                    {String(i + 1)}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{
                      fontFamily: 'var(--mono)', fontSize: 12.5,
                      color: active ? p.color : (done ? 'var(--ink)' : 'rgba(31,26,20,0.4)'),
                      fontWeight: active ? 600 : 500,
                      textTransform: 'uppercase', letterSpacing: '0.06em',
                    }}>
                      {p.label}
                    </div>
                    <div style={{
                      fontFamily: 'var(--serif)', fontSize: 11.5,
                      fontStyle: 'italic',
                      color: active ? 'var(--ink)' : 'var(--ink-soft)',
                      opacity: active ? 1 : done ? 0.65 : 0.4,
                      marginTop: 1,
                    }}>
                      {p.sub}
                    </div>
                  </div>
                  {/* progress hairline on the active phase */}
                  {active && (
                    <div style={{
                      width: 60, height: 2, borderRadius: 1,
                      background: 'rgba(31,26,20,0.1)',
                      position: 'relative', overflow: 'hidden',
                    }}>
                      <div style={{
                        position: 'absolute', left: 0, top: 0,
                        width: `${localProg * 100}%`, height: '100%',
                        background: p.color,
                      }}/>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Tree SVG */}
      <svg style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} width="100%" height="100%">
        {/* Edges: root -> children (always shown faintly once child is born) */}
        {CHILDREN.map((c, i) => {
          if (!childBorn(i)) return null;
          const isOnPath = showRollout && curPath && curPath[1] === i + 1 && simT < 0.6;
          return (
            <g key={`er${i}`}>
              <ArrowCurve
                x1={ROOT.x} y1={ROOT.y + 55}
                x2={c.x} y2={c.y - 5}
                color="var(--ink)"
                strokeWidth={isOnPath ? 2.2 : 1.1}
                opacity={isOnPath ? 1 : 0.55}
                curvature={i === 0 ? 0.12 : i === 2 ? -0.12 : 0}
                progress={1}
              />
            </g>
          );
        })}
        {/* Edges: mid-child -> grandchildren */}
        {GRAND.map((g, i) => {
          if (!grandBorn(i)) return null;
          const parent = CHILDREN[1];
          const isOnPath = showRollout && curPath && curPath[2] === i + 3 && simT < 0.7;
          return (
            <g key={`eg${i}`}>
              <ArrowCurve
                x1={parent.x} y1={parent.y + 55}
                x2={g.x} y2={g.y - 5}
                color="var(--ink)"
                strokeWidth={isOnPath ? 2.2 : 1.1}
                opacity={isOnPath ? 1 : 0.55}
                curvature={i === 0 ? 0.1 : -0.1}
                progress={1}
              />
            </g>
          );
        })}

        {/* PUCT labels on root->child edges (when tweak enabled) */}
        {showPUCT && CHILDREN.map((c, i) => {
          if (!childBorn(i)) return null;
          const n = visitedChildren[i] || 1;
          const q = [0.45, 0.62, 0.38][i];
          const p = [0.28, 0.30, 0.28][i];
          const u = 1.2 * p * Math.sqrt(Math.max(1, simIdx)) / (1 + n);
          const puct = (q + u).toFixed(2);
          return (
            <g key={`pl${i}`}>
              <rect
                x={(ROOT.x + c.x) / 2 - 34}
                y={(ROOT.y + 55 + c.y - 5) / 2 - 10}
                width={68} height={20}
                fill="var(--bg)" stroke="var(--ink-soft)" strokeWidth={0.5}
                rx={10} opacity={0.95}
              />
              <text
                x={(ROOT.x + c.x) / 2}
                y={(ROOT.y + 55 + c.y - 5) / 2 + 4}
                fill="var(--ink)" fontFamily="var(--mono)" fontSize={10}
                textAnchor="middle"
              >
                Q+U={puct}
              </text>
            </g>
          );
        })}

        {/* "Skip the rollout" ghost — during evaluate phase, show the game
            continuation we're NOT playing out, and X it out to emphasize that
            the network value replaces the rest of the game. */}
        {showRollout && simT > 0.48 && simT < 0.75 && curPath && (() => {
          const phaseT = clamp((simT - 0.48) / 0.22, 0, 1);
          const fade = phaseT < 0.6 ? phaseT / 0.6 : clamp(1 - (phaseT - 0.85) / 0.2, 0, 1);
          // Find the leaf we're evaluating (deepest node on curPath)
          const childIdx = curPath[1] ? curPath[1] - 1 : -1;
          const grandIdx = curPath[2] ? curPath[2] - 3 : -1;
          let leaf = null;
          let isGrand = false;
          if (grandIdx >= 0 && grandIdx < 2) { leaf = GRAND[grandIdx]; isGrand = true; }
          else if (childIdx >= 0 && childIdx < 3) leaf = CHILDREN[childIdx];
          if (!leaf) return null;

          // For children (depth 1), ghost goes DOWN. For grandchildren (depth 2,
          // already near canvas bottom), ghost goes to the SIDE.
          const leafSize = isGrand ? 41 : 50; // half-size of the board node
          let dirX, dirY;
          if (isGrand) {
            // fan to the right if leaf is on left, left if on right
            dirX = leaf.x < ROOT.x ? -1 : 1;
            dirY = 0.3;
          } else {
            dirX = 0; dirY = 1;
          }
          const lx = leaf.x + dirX * leafSize;
          const ly = leaf.y + (dirY > 0 ? leafSize : -10);
          const branchLen = isGrand ? 52 : 70;
          const ghostBranches = isGrand
            ? [
                { dx: dirX * branchLen, dy: -24 },
                { dx: dirX * branchLen, dy: 0 },
                { dx: dirX * branchLen, dy: 24 },
              ]
            : [
                { dx: -50, dy: 70 },
                { dx: 0,   dy: 78 },
                { dx: 50,  dy: 70 },
              ];
          return (
            <g opacity={fade}>
              {ghostBranches.map((g, i) => {
                const tx = lx + g.dx;
                const ty = ly + g.dy;
                return (
                  <g key={`gh${i}`}>
                    <line
                      x1={lx} y1={ly}
                      x2={tx} y2={ty}
                      stroke="var(--ink-soft)" strokeWidth={0.9}
                      strokeDasharray="3 3"
                      opacity={0.5}
                    />
                    <circle cx={tx} cy={ty} r={4}
                            fill="none" stroke="var(--ink-soft)"
                            strokeWidth={0.9} opacity={0.4} />
                    {/* second-level ghosts */}
                    {i === 1 && !isGrand && (
                      <>
                        <line x1={tx} y1={ty} x2={tx - 18} y2={ty + 26}
                              stroke="var(--ink-soft)" strokeWidth={0.7}
                              strokeDasharray="2 3" opacity={0.3}/>
                        <line x1={tx} y1={ty} x2={tx + 18} y2={ty + 26}
                              stroke="var(--ink-soft)" strokeWidth={0.7}
                              strokeDasharray="2 3" opacity={0.3}/>
                        <circle cx={tx - 18} cy={ty + 26} r={2.5}
                                fill="none" stroke="var(--ink-soft)"
                                strokeWidth={0.7} opacity={0.25} />
                        <circle cx={tx + 18} cy={ty + 26} r={2.5}
                                fill="none" stroke="var(--ink-soft)"
                                strokeWidth={0.7} opacity={0.25} />
                      </>
                    )}
                  </g>
                );
              })}

              {/* X through the ghost subtree */}
              {phaseT > 0.25 && (() => {
                const xProg = clamp((phaseT - 0.25) / 0.35, 0, 1);
                const cx = isGrand ? lx + dirX * (branchLen - 10) : lx;
                const cy = isGrand ? ly : ly + 54;
                const s = isGrand ? 30 : 44;
                return (
                  <g>
                    <line
                      x1={cx - s} y1={cy - s * 0.55}
                      x2={cx - s + (2 * s) * xProg} y2={cy - s * 0.55 + (s * 1.1) * xProg}
                      stroke="var(--accent-mcts)" strokeWidth={2.2}
                      strokeLinecap="round"
                    />
                    {xProg > 0.5 && (
                      <line
                        x1={cx + s} y1={cy - s * 0.55}
                        x2={cx + s - (2 * s) * clamp((xProg - 0.5) / 0.5, 0, 1)}
                        y2={cy - s * 0.55 + (s * 1.1) * clamp((xProg - 0.5) / 0.5, 0, 1)}
                        stroke="var(--accent-mcts)" strokeWidth={2.2}
                        strokeLinecap="round"
                      />
                    )}
                  </g>
                );
              })()}

              {/* Callout label — always placed safely to the right of the action */}
              {phaseT > 0.45 && (() => {
                const labelOp = clamp((phaseT - 0.45) / 0.3, 0, 1);
                const labelX = isGrand
                  ? (dirX < 0 ? leaf.x - 220 : leaf.x + 90)
                  : Math.min(leaf.x + 110, 940);
                const labelY = isGrand ? leaf.y - 40 : leaf.y - 20;
                return (
                  <g opacity={labelOp}>
                    <rect
                      x={labelX} y={labelY}
                      width={210} height={48}
                      rx={6}
                      fill="var(--bg)"
                      stroke="var(--accent-net)" strokeWidth={1}
                    />
                    <text
                      x={labelX + 10} y={labelY + 17}
                      fontFamily="var(--mono)" fontSize={10}
                      letterSpacing="0.1em"
                      fill="var(--accent-net)"
                      style={{ textTransform: 'uppercase' }}
                    >
                      value head — no rollout
                    </text>
                    <text
                      x={labelX + 10} y={labelY + 34}
                      fontFamily="var(--serif)" fontSize={12} fontStyle="italic"
                      fill="var(--ink)"
                    >
                      v̂ = f<tspan baselineShift="sub" fontSize={9}>θ</tspan>(s) estimates the outcome
                    </text>
                  </g>
                );
              })()}
            </g>
          );
        })()}

        {/* Backup pulse along the current path (when simT > 0.7) */}
        {showRollout && simT > 0.7 && curPath && (() => {
          const childIdx = curPath[1] ? curPath[1] - 1 : -1;
          const grandIdx = curPath[2] ? curPath[2] - 3 : -1;
          const childNode = childIdx >= 0 && childIdx < 3 ? CHILDREN[childIdx] : null;
          const grandNode = grandIdx >= 0 && grandIdx < 2 ? GRAND[grandIdx] : null;
          const nodes = [ROOT, ...(childNode ? [childNode] : []), ...(grandNode ? [grandNode] : [])];
          if (nodes.length < 2) return null;
          const pulseT = (simT - 0.7) / 0.3;
          const segIdx = Math.floor(pulseT * (nodes.length - 1));
          const segT = (pulseT * (nodes.length - 1)) - segIdx;
          if (segIdx >= nodes.length - 1) return null;
          const a = nodes[nodes.length - 1 - segIdx];
          const b = nodes[nodes.length - 2 - segIdx];
          const x = a.x + (b.x - a.x) * segT;
          const y = (a.y + 55) + ((b.y - 5) - (a.y + 55)) * segT;
          return (
            <circle cx={x} cy={y} r={6} fill="var(--accent-mcts)" opacity={0.9}>
              <animate attributeName="r" values="4;8;4" dur="0.5s" repeatCount="indefinite" />
            </circle>
          );
        })()}
      </svg>

      {/* Root node */}
      <TreeNode
        cx={ROOT.x} cy={ROOT.y}
        stones={ROOT_STONES}
        visits={simIdx}
        color="var(--ink)"
        size={110}
        labelAbove="root"
        highlight={showRollout && simT < 0.2}
      />

      {/* Children */}
      {CHILDREN.map((c, i) => {
        if (!childBorn(i)) return null;
        let birthSim = -1;
        for (let s = 0; s < simCount; s++) {
          if (PATHS[s] && PATHS[s][1] === i + 1) { birthSim = s; break; }
        }
        const isCurrent = showRollout && curPath && curPath[1] === i + 1;
        const entry = birthSim === simIdx && showRollout
          ? Easing.easeOutBack(clamp((simT - 0.3) / 0.25, 0, 1))
          : 1;
        const extraMove = [{ x: [0, 1, 2][i], y: 1, color: 'W' }];
        return (
          <TreeNode
            key={`c${i}`}
            cx={c.x} cy={c.y}
            stones={[...ROOT_STONES, ...extraMove]}
            highlights={[{ x: [0, 1, 2][i], y: 1 }]}
            visits={visitedChildren[i]}
            valueShown={birthSim <= simIdx && showRollout && isCurrent && simT > 0.5}
            valueText={`v=${[0.38, 0.72, 0.55][i].toFixed(2)}`}
            netShown={birthSim <= simIdx - 1 || (birthSim === simIdx && showRollout && simT > 0.4)}
            color="var(--ink)"
            size={100}
            scale={entry}
            highlight={isCurrent && simT >= 0.2 && simT < 0.5}
          />
        );
      })}

      {/* Grandchildren */}
      {GRAND.map((g, i) => {
        if (!grandBorn(i)) return null;
        let birthSim = -1;
        for (let s = 0; s < simCount; s++) {
          if (PATHS[s] && PATHS[s].includes(i + 3)) { birthSim = s; break; }
        }
        const isCurrent = showRollout && curPath && curPath[2] === i + 3;
        const entry = birthSim === simIdx && showRollout
          ? Easing.easeOutBack(clamp((simT - 0.35) / 0.25, 0, 1))
          : 1;
        const gStones = [
          ...ROOT_STONES,
          { x: 1, y: 1, color: 'W' }, // mid move
          { x: i === 0 ? 0 : 2, y: 1, color: 'B' },
        ];
        return (
          <TreeNode
            key={`g${i}`}
            cx={g.x} cy={g.y}
            stones={gStones}
            highlights={[{ x: i === 0 ? 0 : 2, y: 1, color: 'var(--accent-mcts)' }]}
            visits={visitedGrand[i]}
            valueShown={birthSim <= simIdx && showRollout && isCurrent && simT > 0.6}
            valueText={`v=${[0.55, 0.38][i].toFixed(2)}`}
            color="var(--ink-soft)"
            size={82}
            scale={entry}
            highlight={isCurrent && simT >= 0.35 && simT < 0.6}
          />
        );
      })}
    </>
  );
}

// ── TreeNode ────────────────────────────────────────────────────────────────
function TreeNode({
  cx, cy, stones = [], highlights = [], visits = 0,
  valueShown = false, valueText = '', netShown = false,
  color = 'var(--ink)', size = 100, scale = 1,
  highlight = false, labelAbove = null,
}) {
  return (
    <div style={{
      position: 'absolute',
      left: cx - size / 2,
      top: cy - size / 2,
      transform: `scale(${scale})`,
      transformOrigin: 'center',
    }}>
      <div style={{ position: 'relative' }}>
        {labelAbove && (
          <div style={{
            position: 'absolute', bottom: '100%', left: '50%',
            transform: 'translateX(-50%)',
            fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--ink-soft)',
            marginBottom: 4,
          }}>{labelAbove}</div>
        )}
        <div style={{
          outline: highlight ? '2px solid var(--accent-mcts)' : 'none',
          outlineOffset: 3,
          borderRadius: 16,
          transition: 'outline 160ms',
        }}>
          <GoBoard
            n={3}
            size={size}
            stones={stones}
            highlights={highlights}
            radius={12}
          />
        </div>
        {/* visit count badge */}
        {visits > 0 && (
          <div style={{
            position: 'absolute',
            top: -6, right: -8,
            minWidth: 22, height: 22,
            padding: '0 6px',
            background: 'var(--ink)',
            color: 'var(--bg)',
            borderRadius: 11,
            fontFamily: 'var(--mono)',
            fontSize: 11,
            fontWeight: 500,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 2px 4px rgba(0,0,0,0.12)',
          }}>
            N={visits}
          </div>
        )}
        {/* net glyph below */}
        {netShown && (
          <div style={{
            position: 'absolute', top: '100%', left: '50%',
            transform: 'translateX(-50%)',
            marginTop: 6,
            display: 'flex', flexDirection: 'column', alignItems: 'center',
          }}>
            <NetGlyph width={40} height={22} color="var(--ink-soft)" />
            {valueShown && (
              <div style={{
                fontFamily: 'var(--mono)', fontSize: 10.5,
                color: 'var(--accent-mcts)', marginTop: 3,
              }}>
                {valueText}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Act 2c: PUCT formula — UCB1 → PUCT ──────────────────────────────────────
// Inserted between "03 · mcts" (tree walk) and "04 · pseudocode". Builds the
// selection rule from its bandit ancestor (UCB1) and shows why a tree with a
// huge branching factor needs the prior-weighted form.
function Act2c_PUCT() {
  const { localTime: lt } = useSprite();
  const op = (at, dur = 0.6) => Easing.easeOutCubic(clamp((lt - at) / dur, 0, 1));

  const introOp   = op(0.0);
  const ucbOp     = op(0.6);
  const bridgeOp  = op(1.6);
  const puctOp    = op(2.4);
  const noteOp    = op(4.4);

  const FORMULA_BOX = {
    fontFamily: 'var(--mono)', fontSize: 16,
    padding: '14px 18px',
    background: 'var(--accent-mcts-bg)',
    border: '1px solid var(--accent-mcts)',
    borderRadius: 10,
    color: 'var(--accent-mcts)',
    display: 'inline-block',
    lineHeight: 1.4,
  };

  return (
    <>
      <SlideHeader
        num="04"
        title="From UCB1 to PUCT"
        maxWidth={820}
        subtitle={<>
          UCB1 is the classic multi-armed-bandit rule: try every arm at least once, then favour arms whose confidence
          interval still leaves room above the best mean. PUCT is its tree-search cousin — modified for the case where
          we <em>can't</em> consider every arm even once.
        </>}
      />

      {/* UCB1 block */}
      <div style={{
        position: 'absolute', left: 200, top: 230,
        opacity: ucbOp, transform: `translateY(${(1 - ucbOp) * 6}px)`,
      }}>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 11,
          letterSpacing: '0.14em', textTransform: 'uppercase',
          color: 'var(--ink-soft)', marginBottom: 8,
        }}>
          UCB1 · multi-armed bandits
        </div>
        <div style={FORMULA_BOX}>
          a<sub>t</sub> = argmax<sub>a</sub> [ Q(a) + √( 2·ln N ⁄ N(a) ) ]
        </div>

        <div style={{
          marginTop: 12,
          display: 'flex', flexDirection: 'column', gap: 6,
          fontFamily: 'var(--serif)', fontSize: 12.5,
          color: 'var(--ink-soft)', lineHeight: 1.5,
          maxWidth: 540,
        }}>
          <div>
            <span style={{ fontFamily: 'var(--mono)', color: 'var(--accent-mcts)', fontWeight: 600 }}>
              Q(a)
            </span>
            <span style={{ margin: '0 8px', opacity: 0.5 }}>—</span>
            empirical mean reward.
          </div>
          <div>
            <span style={{ fontFamily: 'var(--mono)', color: 'var(--accent-mcts)', fontWeight: 600 }}>
              √( 2·ln N ⁄ N(a) )
            </span>
            <span style={{ margin: '0 8px', opacity: 0.5 }}>—</span>
            exploration bonus; shrinks as <span style={{ fontFamily: 'var(--mono)' }}>n<sub>i</sub></span> grows.
            The more times we select an action, the higher <span style={{ fontFamily: 'var(--mono)' }}>n<sub>i</sub></span>{' '}
            grows, which decreases its bonus.
          </div>
        </div>
      </div>

      {/* PUCT block — single combined formula */}
      <div style={{
        position: 'absolute', left: 200, top: 470,
        opacity: puctOp, transform: `translateY(${(1 - puctOp) * 6}px)`,
        maxWidth: 640,
      }}>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 11,
          letterSpacing: '0.14em', textTransform: 'uppercase',
          color: 'var(--accent-mcts)', marginBottom: 8,
        }}>
          PUCT · selection rule
        </div>
        <div style={{ ...FORMULA_BOX, fontSize: 14.5 }}>
          a<sub>t</sub> = argmax<sub>a</sub> [ Q(s,a) + c<sub>puct</sub> · P(s,a) · √( N ) ⁄ ( 1 + N(a) ) ]
        </div>

        <div style={{
          marginTop: 12,
          fontFamily: 'var(--serif)', fontSize: 12.5,
          color: 'var(--ink-soft)', lineHeight: 1.5,
        }}>
          The exploration factor{' '}
          <span style={{ fontFamily: 'var(--mono)', color: 'var(--accent-mcts)', fontWeight: 600 }}>
            √(N) ⁄ (1 + N(a))
          </span>{' '}
          shrinks slower than UCB1's{' '}
          <span style={{ fontFamily: 'var(--mono)' }}>√(2·ln t ⁄ n<sub>a</sub>)</span> as total visits accumulate —
          <span style={{ fontFamily: 'var(--mono)' }}> √N</span> grows faster than{' '}
          <span style={{ fontFamily: 'var(--mono)' }}>√(ln t)</span>, so the bonus for less-visited actions stays
          large longer and PUCT keeps exploring instead of locking in.
        </div>
      </div>

    </>
  );
}

// ── Act 3: Visit counts → π* ────────────────────────────────────────────────
// 17..23s. Shrink tree to the side, show visit histogram, normalize to π*.
function Act3_NormalizeTarget() {
  const t = useTime();
  const { localTime: lt } = useSprite();

  // Tree thumbnail on left
  // Big bars labelled N(s,·) morphing from raw counts to normalized π*

  const morphT = clamp((lt - 2.0) / 1.5, 0, 1); // visits → π*
  const barsOn = clamp(lt / 0.5, 0, 1);

  // Left prior, right target
  const maxV = Math.max(...VISITS);
  const rawNorm = VISITS.map(v => v / maxV); // all relative to biggest
  const piStar = VISITS.map(v => v / VISITS.reduce((a, b) => a + b, 0));
  // For display, scale both to same pixel height
  const displayed = rawNorm.map((v, i) => v + (piStar[i] * (Math.max(...VISITS) / VISITS.reduce((a, b) => a + b, 0)) - v) * morphT);
  // simpler: show bars always relative to max — then label changes
  const displayBars = VISITS.map(v => v / maxV);

  return (
    <>
      <div style={{ position: 'absolute', left: 200, top: 100, opacity: clamp(lt / 0.3, 0, 1) }}>
        <SectionLabel num="06" title="MCTS visit counts become the policy target π*" />
      </div>

      <div style={{ position: 'absolute', left: 200, top: 180, maxWidth: 400, opacity: clamp((lt - 0.3) / 0.5, 0, 1) }}>
        <Caption size={14} style={{ lineHeight: 1.55 }}>
          After a few hundred simulations, the move that was explored most is the move the search thought was the best. Normalise the visit counts to obtain an "improved policy target"
        </Caption>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--accent-mcts)',
          marginTop: 14, padding: '10px 14px',
          background: 'var(--accent-mcts-bg)',
          borderRadius: 8,
        }}>
          π*(a | s) = N(s, a)<sup>1/τ</sup> / Σ N(s, b)<sup>1/τ</sup>
        </div>
        <Caption size={13} style={{ marginTop: 12, lineHeight: 1.5 }}>
          This "action relabeling" is applied to all states, regardless of win or lose.
          This is analogous to{' '}
          <a href="https://arxiv.org/abs/1011.0686"
             target="_blank"
             rel="noopener noreferrer"
             onClick={(e) => e.stopPropagation()}
             style={{
               color: 'var(--accent-mcts)',
               textDecoration: 'underline',
               textUnderlineOffset: 2,
               pointerEvents: 'auto',
             }}>
            DAgger
          </a>.
        </Caption>
      </div>


      {/* Big bars */}
      <div style={{
        position: 'absolute', right: 120, top: 240,
        opacity: barsOn,
        transform: `translateY(${(1 - Easing.easeOutCubic(barsOn)) * 16}px)`,
      }}>
        <div style={{
          padding: '24px 28px',
          background: '#ffffff55',
          border: `1px solid var(--accent-mcts)`,
          borderRadius: 14,
        }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 14, height: 220 }}>
            {displayBars.map((v, i) => {
              const h = v * 200;
              const isMax = i === 5;
              const labelVisit = VISITS[i];
              const labelPi = piStar[i].toFixed(2);
              return (
                <div key={i} style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'center',
                  width: 36,
                }}>
                  <div style={{
                    fontFamily: 'var(--mono)', fontSize: 11,
                    color: isMax ? 'var(--accent-mcts)' : 'var(--ink-soft)',
                    marginBottom: 6, fontWeight: isMax ? 600 : 400,
                    height: 14,
                  }}>
                    {morphT < 0.5 ? `N=${labelVisit}` : labelPi}
                  </div>
                  <div style={{
                    width: 32, height: h,
                    background: 'var(--accent-mcts)',
                    borderRadius: '2px 2px 0 0',
                    opacity: 0.5 + 0.5 * (v),
                  }}/>
                </div>
              );
            })}
          </div>
          <Caption mono size={12} color="var(--accent-mcts)" style={{ marginTop: 14, textAlign: 'center' }}>
            {morphT < 0.5
              ? 'visit counts N(s, a)'
              : 'π*(· | s) — the supervision target'}
          </Caption>
        </div>
      </div>

      {/* Small prior bars below-left for comparison (once morphT > 0.5) */}
      {morphT > 0.4 && (
        <div style={{
          position: 'absolute', left: 200, bottom: 80,
          opacity: clamp((morphT - 0.4) / 0.3, 0, 1),
          display: 'flex', alignItems: 'center', gap: 20,
        }}>
          <div>
            <PolicyBars values={PRIOR} width={120} height={42} color="var(--accent-net)" />
            <Caption mono size={11} color="var(--accent-net)" style={{ marginTop: 4 }}>
              prior p (network)
            </Caption>
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 18, color: 'var(--ink-soft)' }}>vs</div>
          <div>
            <PolicyBars values={TARGET} width={120} height={42} color="var(--accent-mcts)" />
            <Caption mono size={11} color="var(--accent-mcts)" style={{ marginTop: 4 }}>
              π* (MCTS, sharper)
            </Caption>
          </div>
        </div>
      )}
    </>
  );
}

// ── Act 4: Network chases target ────────────────────────────────────────────
// 23..32s. Bars morph + KL divergence callout (the actual learning signal).
function Act4_NetworkChasesTarget() {
  const t = useTime();
  const { localTime: lt } = useSprite();

  // Morph t from PRIOR toward TARGET but not all the way (partial update — SGD step)
  const chase = Easing.easeInOutCubic(clamp((lt - 1.2) / 2.5, 0, 1)) * 0.7;
  const current = lerpArr(PRIOR, TARGET, chase);

  // KL across iterations (the learning signal over training)
  // Real data vibes: starts ~0.5 nats, decays with noise, different opponents
  const KL_ITERS = 17;
  const klCurve = (iter, opp) => {
    // opp: 0..5, different decay rates and baselines
    const bases = [0.35, 0.48, 0.30, 0.42, 0.55, 0.28];
    const decays = [3.0, 4.0, 2.6, 3.4, 5.0, 2.4];
    const noise = [0.02, 0.03, 0.02, 0.04, 0.06, 0.015];
    const base = bases[opp] * Math.exp(-iter / decays[opp]);
    const n = noise[opp] * Math.sin(iter * (1.3 + opp * 0.2) + opp);
    return Math.max(0.005, base + n);
  };
  const OPPS = [
    { name: 'katago-2024-02', color: 'oklch(62% 0.12 245)' },
    { name: 'katago-2025-03', color: 'oklch(68% 0.10 140)' },
    { name: 'katago-2026-03', color: 'oklch(72% 0.10 80)' },
    { name: 'katago-2026-04', color: 'oklch(65% 0.14 30)' },
    { name: 'human',          color: 'oklch(55% 0.18 320)' },
    { name: 'self-play',      color: 'oklch(60% 0.12 180)' },
  ];

  // KL chart reveal starts ~lt 4.0
  const klReveal = clamp((lt - 4.0) / 3.5, 0, 1);
  const klMax = Math.max(0, Math.floor(klReveal * (KL_ITERS - 1) + 0.001));

  return (
    <>
      <div style={{ position: 'absolute', left: 200, top: 80, opacity: clamp(lt / 0.3, 0, 1) }}>
        <SectionLabel num="07" title="Train the network to match π*" />
      </div>

      <div style={{ position: 'absolute', left: 200, top: 160, maxWidth: 430, opacity: clamp((lt - 0.4) / 0.5, 0, 1) }}>
        <Caption size={14} style={{ lineHeight: 1.55 }}>
          A gradient step minimises the cross-entropy between the network's prior and the
          MCTS target. The prior <em>shifts</em> toward π*, but doesn't reach it — the next
          search, informed by the better prior, will produce an even sharper target.
        </Caption>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--ink)',
          marginTop: 12, padding: '10px 14px',
          background: 'rgba(0,0,0,0.04)', borderRadius: 8,
        }}>
          L = −Σ π*(a) · log π<sub>θ</sub>(a | s)
        </div>
      </div>

      {/* Two bar charts side-by-side */}
      <div style={{
        position: 'absolute', right: 120, top: 140,
        display: 'flex', alignItems: 'flex-end', gap: 32,
      }}>
        {/* Target */}
        <div>
          <div style={{
            padding: '16px 20px',
            background: 'rgba(255,255,255,0.4)',
            border: '1px solid var(--accent-mcts)',
            borderRadius: 12,
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 7, height: 130 }}>
              {TARGET.map((v, i) => (
                <div key={i} style={{
                  width: 20, height: v * 120 + 3,
                  background: 'var(--accent-mcts)',
                  borderRadius: '1px 1px 0 0',
                }}/>
              ))}
            </div>
          </div>
          <Caption mono size={11} color="var(--accent-mcts)" style={{ marginTop: 8, textAlign: 'center' }}>
            π* target
          </Caption>
        </div>

        {/* Arrow */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, marginBottom: 30 }}>
          <svg width={48} height={28}>
            <path d="M 4 14 Q 24 4 44 14" stroke="var(--ink)" strokeWidth={1.4} fill="none"
                  strokeLinecap="round" markerEnd="url(#arrhead-chase)" />
            <defs>
              <marker id="arrhead-chase" viewBox="0 0 10 10" refX={8} refY={5}
                      markerWidth={5} markerHeight={5} orient="auto">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--ink)" />
              </marker>
            </defs>
          </svg>
          <Caption mono size={10} color="var(--ink-soft)">
            grad
          </Caption>
        </div>

        {/* Current policy */}
        <div>
          <div style={{
            padding: '16px 20px',
            background: 'rgba(255,255,255,0.4)',
            border: '1px solid var(--accent-net)',
            borderRadius: 12,
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 7, height: 130 }}>
              {current.map((v, i) => (
                <div key={i} style={{
                  width: 20,
                  height: v * 120 + 3,
                  background: 'var(--accent-net)',
                  borderRadius: '1px 1px 0 0',
                  transition: 'height 180ms ease-out',
                }}/>
              ))}
            </div>
          </div>
          <Caption mono size={11} color="var(--accent-net)" style={{ marginTop: 8, textAlign: 'center' }}>
            π<sub>θ</sub> (training ↗)
          </Caption>
        </div>
      </div>

      {/* KL divergence insight panel */}
      {lt > 3.6 && (
        <div style={{
          position: 'absolute', left: 200, bottom: 70, right: 80,
          opacity: clamp((lt - 3.6) / 0.6, 0, 1),
          transform: `translateY(${(1 - clamp((lt - 3.6) / 0.6, 0, 1)) * 14}px)`,
          borderTop: '1px solid rgba(31,26,20,0.12)',
          paddingTop: 20,
          display: 'grid',
          gridTemplateColumns: '340px 1fr',
          gap: 40,
        }}>
          {/* Left: explainer */}
          <div>
            <div style={{
              fontFamily: 'var(--mono)', fontSize: 11,
              letterSpacing: '0.12em', textTransform: 'uppercase',
              color: 'var(--accent-mcts)', marginBottom: 8,
            }}>
              The signal itself
            </div>
            <div style={{
              fontFamily: 'var(--serif)', fontSize: 16,
              color: 'var(--ink)', lineHeight: 1.45, marginBottom: 10,
              letterSpacing: '-0.005em',
            }}>
              KL(π* ‖ π<sub>θ</sub>) <em>is</em> the learning signal for the next iteration.
            </div>
            <Caption size={12.5} style={{ lineHeight: 1.5 }}>
              Measured on live games against many opponents, the divergence between MCTS and
              the network shrinks as the network absorbs the search's sharper distributions.
              When the gap flattens, training has converged for that opponent distribution.
            </Caption>
          </div>

          {/* Right: small-multiples chart */}
          <div>
            <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 6 }}>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink-soft)' }}>
                KL(MCTS ‖ π<sub>θ</sub>) <span style={{ opacity: 0.6 }}>nats</span>
              </div>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                {OPPS.map((o, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4,
                                         fontFamily: 'var(--mono)', fontSize: 9.5,
                                         color: 'var(--ink-soft)' }}>
                    <span style={{ width: 8, height: 8, background: o.color, borderRadius: 2 }} />
                    {o.name}
                  </div>
                ))}
              </div>
            </div>
            <KLChart
              width={720} height={150}
              iters={KL_ITERS}
              revealIdx={klReveal * (KL_ITERS - 1)}
              opps={OPPS}
              klFn={klCurve}
            />
            <div style={{
              display: 'flex', justifyContent: 'space-between',
              fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--ink-soft)',
              marginTop: 4, paddingLeft: 28, paddingRight: 4,
            }}>
              <span>iter 0</span>
              <span style={{ fontStyle: 'italic' }}>training iterations →</span>
              <span>iter {KL_ITERS - 1}</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// ── KLChart ────────────────────────────────────────────────────────────────
// Small-multiples-feel chart: one bar per (iter, opponent), grouped by iter.
function KLChart({ width, height, iters, revealIdx, opps, klFn }) {
  const left = 28, right = 6, top = 6, bottom = 4;
  const chartW = width - left - right;
  const chartH = height - top - bottom;
  const groupW = chartW / iters;
  const barW = groupW / (opps.length + 0.6);
  const yMax = 0.8;

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      {/* y-axis gridlines */}
      {[0.2, 0.4, 0.6, 0.8].map(v => {
        const y = top + (1 - v / yMax) * chartH;
        return (
          <g key={v}>
            <line x1={left} y1={y} x2={left + chartW} y2={y}
                  stroke="var(--ink-soft)" strokeWidth={0.4} opacity={0.18}
                  strokeDasharray="2 3" />
            <text x={left - 4} y={y + 3} textAnchor="end"
                  fill="var(--ink-soft)" fontFamily="var(--mono)" fontSize={9}>
              {v.toFixed(1)}
            </text>
          </g>
        );
      })}
      {/* baseline */}
      <line x1={left} y1={top + chartH} x2={left + chartW} y2={top + chartH}
            stroke="var(--ink-soft)" strokeWidth={0.6} opacity={0.5} />

      {/* bars */}
      {Array.from({ length: iters }).map((_, i) => {
        const revealN = clamp(revealIdx - i + 1, 0, 1); // 0..1 within this iter
        if (revealN <= 0) return null;
        const gx = left + i * groupW + groupW * 0.1;
        return (
          <g key={i}>
            {opps.map((opp, j) => {
              const v = klFn(i, j);
              const h = (v / yMax) * chartH * revealN;
              return (
                <rect
                  key={j}
                  x={gx + j * barW}
                  y={top + chartH - h}
                  width={barW * 0.9}
                  height={Math.max(0.5, h)}
                  fill={opp.color}
                  opacity={0.85}
                />
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

// ── Act 5: Outer loop — moving target over iterations ───────────────────────
// 28..36s. Skill-axis chart: MCTS line above π_net line, both climb; gap narrows at top.
function Act5_MovingTarget() {
  const { localTime: lt, duration } = useSprite();

  // Chart bounds — left margin matches the rest of the deck (200) so the
  // plot doesn't slide under the left-hand sidebar.
  const chartX = 200, chartY = 270;
  const chartW = 720, chartH = 300;

  // Iteration dot positions (x = iter, y = skill)
  // We'll show 10 iterations. MCTS always ~12% above π_net, both asymptoting.
  const N = 10;
  const netSkill = (i) => 0.1 + 0.85 * (1 - Math.exp(-i / 3.0));
  const mctsSkill = (i) => Math.min(0.98, netSkill(i) + 0.14 * Math.exp(-i / 6.0) + 0.02);

  const xOf = (i) => chartX + (i / (N - 1)) * chartW;
  const yOf = (s) => chartY + chartH - s * chartH;

  // Reveal progress along the curves
  const reveal = clamp((lt - 0.8) / 4.5, 0, 1); // 0..1 over 4.5s
  const revealIdx = reveal * (N - 1);

  // Partial point list for a polyline up to revealIdx
  const pointsUpTo = (fn) => {
    const pts = [];
    for (let i = 0; i <= Math.floor(revealIdx); i++) pts.push([xOf(i), yOf(fn(i))]);
    // Last partial segment
    const frac = revealIdx - Math.floor(revealIdx);
    if (frac > 0 && Math.floor(revealIdx) + 1 < N) {
      const a = Math.floor(revealIdx);
      const b = a + 1;
      const x = xOf(a) + (xOf(b) - xOf(a)) * frac;
      const y = yOf(fn(a)) + (yOf(fn(b)) - yOf(fn(a))) * frac;
      pts.push([x, y]);
    }
    return pts.map(p => p.join(',')).join(' ');
  };

  // Chasing dots (current heads of each curve)
  const curIdx = Math.min(N - 1, revealIdx);
  const netHead = { x: xOf(curIdx), y: yOf(netSkill(curIdx)) };
  const mctsHead = { x: xOf(curIdx), y: yOf(mctsSkill(curIdx)) };

  return (
    <>
      <div style={{ position: 'absolute', left: 200, top: 100, opacity: clamp(lt / 0.3, 0, 1) }}>
        <SectionLabel num="08 / Moving target" title="Iterate: the target always stays one step ahead" />
      </div>

      <div style={{ position: 'absolute', left: 200, top: 170, maxWidth: 720, opacity: clamp((lt - 0.3) / 0.5, 0, 1) }}>
        <Caption size={14} style={{ lineHeight: 1.55 }}>
          Because MCTS <em>uses</em> the current network and then improves on it by searching, its
          recommendations are always stronger than the network alone. The gap powers training; as
          the network gets stronger, the ceiling of what search can uncover rises with it.
        </Caption>
      </div>

      <svg style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} width="100%" height="100%">
        {/* Axes */}
        <line x1={chartX} y1={chartY + chartH} x2={chartX + chartW} y2={chartY + chartH}
              stroke="var(--ink-soft)" strokeWidth={0.8} opacity={0.5} />
        <line x1={chartX} y1={chartY} x2={chartX} y2={chartY + chartH}
              stroke="var(--ink-soft)" strokeWidth={0.8} opacity={0.5} />

        {/* Gridlines */}
        {[0.25, 0.5, 0.75].map(s => (
          <line key={s} x1={chartX} y1={yOf(s)} x2={chartX + chartW} y2={yOf(s)}
                stroke="var(--ink-soft)" strokeWidth={0.5} opacity={0.15} strokeDasharray="2 3" />
        ))}

        {/* Optimal ceiling */}
        <line x1={chartX} y1={yOf(1.0)} x2={chartX + chartW} y2={yOf(1.0)}
              stroke="var(--accent-mcts)" strokeWidth={0.8} opacity={0.35} strokeDasharray="4 3" />
        <text x={chartX + 8} y={yOf(1.0) - 6}
              fill="var(--accent-mcts)" fontFamily="var(--mono)" fontSize={11}
              textAnchor="start" opacity={0.7}>
          optimal policy π★
        </text>

        {/* MCTS curve (target) */}
        <polyline
          points={pointsUpTo(mctsSkill)}
          fill="none"
          stroke="var(--accent-mcts)"
          strokeWidth={2.2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* Network curve (chasing) */}
        <polyline
          points={pointsUpTo(netSkill)}
          fill="none"
          stroke="var(--accent-net)"
          strokeWidth={2.2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Shaded gap between them */}
        {reveal > 0.05 && (() => {
          const topPts = [];
          const botPts = [];
          const maxI = Math.floor(revealIdx);
          for (let i = 0; i <= maxI; i++) {
            topPts.push(`${xOf(i)},${yOf(mctsSkill(i))}`);
            botPts.push(`${xOf(i)},${yOf(netSkill(i))}`);
          }
          const frac = revealIdx - maxI;
          if (frac > 0 && maxI + 1 < N) {
            const a = maxI, b = a + 1;
            const x = xOf(a) + (xOf(b) - xOf(a)) * frac;
            topPts.push(`${x},${yOf(mctsSkill(a) + (mctsSkill(b) - mctsSkill(a)) * frac)}`);
            botPts.push(`${x},${yOf(netSkill(a) + (netSkill(b) - netSkill(a)) * frac)}`);
          }
          const poly = [...topPts, ...botPts.reverse()].join(' ');
          return <polygon points={poly} fill="var(--accent-mcts)" opacity={0.08} />;
        })()}

        {/* Iteration dots on each curve */}
        {Array.from({ length: Math.min(N, Math.floor(revealIdx) + 1) }).map((_, i) => (
          <g key={i}>
            <circle cx={xOf(i)} cy={yOf(mctsSkill(i))} r={4.5}
                    fill="var(--accent-mcts)" stroke="var(--bg)" strokeWidth={1.5} />
            <circle cx={xOf(i)} cy={yOf(netSkill(i))} r={4.5}
                    fill="var(--accent-net)" stroke="var(--bg)" strokeWidth={1.5} />
          </g>
        ))}

        {/* Head pulses */}
        {reveal < 1 && (
          <>
            <circle cx={mctsHead.x} cy={mctsHead.y} r={8}
                    fill="var(--accent-mcts)" opacity={0.35}>
              <animate attributeName="r" values="5;12;5" dur="1.2s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.4;0;0.4" dur="1.2s" repeatCount="indefinite" />
            </circle>
            <circle cx={netHead.x} cy={netHead.y} r={8}
                    fill="var(--accent-net)" opacity={0.35}>
              <animate attributeName="r" values="5;12;5" dur="1.2s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.4;0;0.4" dur="1.2s" repeatCount="indefinite" />
            </circle>
          </>
        )}

        {/* Curve labels */}
        {reveal > 0.35 && (
          <g opacity={clamp((reveal - 0.3) / 0.2, 0, 1)}>
            <text x={xOf(N - 1) + 12} y={yOf(mctsSkill(N - 1)) + 4}
                  fill="var(--accent-mcts)" fontFamily="var(--mono)" fontSize={12}
                  fontWeight={600}>
              π* MCTS
            </text>
            <text x={xOf(N - 1) + 12} y={yOf(netSkill(N - 1)) + 4}
                  fill="var(--accent-net)" fontFamily="var(--mono)" fontSize={12}
                  fontWeight={600}>
              πθ net
            </text>
          </g>
        )}

        {/* Axis labels */}
        <text x={chartX + chartW / 2} y={chartY + chartH + 34}
              fill="var(--ink-soft)" fontFamily="var(--serif)" fontSize={13}
              textAnchor="middle" fontStyle="italic">
          training iterations →
        </text>
        <text x={chartX - 12} y={chartY + chartH / 2}
              fill="var(--ink-soft)" fontFamily="var(--serif)" fontSize={13}
              textAnchor="middle" fontStyle="italic"
              transform={`rotate(-90 ${chartX - 12} ${chartY + chartH / 2})`}>
          playing strength →
        </text>

        {/* Small "gap" label */}
        {reveal > 0.2 && (() => {
          const i = Math.min(3, Math.floor(revealIdx));
          const midY = (yOf(mctsSkill(i)) + yOf(netSkill(i))) / 2;
          return (
            <g opacity={clamp((reveal - 0.2) / 0.2, 0, 1) * (reveal > 0.75 ? (1 - (reveal - 0.75) / 0.25) : 1)}>
              <line x1={xOf(i)} y1={yOf(mctsSkill(i)) + 4}
                    x2={xOf(i)} y2={yOf(netSkill(i)) - 4}
                    stroke="var(--ink-soft)" strokeWidth={0.8}
                    markerStart="url(#gap-ar)" markerEnd="url(#gap-ar2)" />
              <defs>
                <marker id="gap-ar" viewBox="0 0 10 10" refX={5} refY={5} markerWidth={5} markerHeight={5} orient="auto">
                  <path d="M5,0 L0,5 L5,10 z" fill="var(--ink-soft)" />
                </marker>
                <marker id="gap-ar2" viewBox="0 0 10 10" refX={5} refY={5} markerWidth={5} markerHeight={5} orient="auto">
                  <path d="M0,0 L5,5 L0,10 z" fill="var(--ink-soft)" />
                </marker>
              </defs>
              <rect x={xOf(i) + 8} y={midY - 14} width={120} height={26}
                    rx={13} fill="var(--bg)" stroke="var(--ink-soft)" strokeWidth={0.5} />
              <text x={xOf(i) + 68} y={midY + 4} textAnchor="middle"
                    fill="var(--ink)" fontFamily="var(--mono)" fontSize={11}>
                training signal
              </text>
            </g>
          );
        })()}
      </svg>
    </>
  );
}

Object.assign(window, {
  HeroTitle, Act1_PriorFromNet, Act2_TreeSearch, Act2c_PUCT, Act3_NormalizeTarget,
  Act4_NetworkChasesTarget, Act5_MovingTarget,
});
