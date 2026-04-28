// components/act3b.jsx
// Act 3b — Misconception: we're not imitating the winner.
//
// Big idea: in self-play, MCTS search is what produces improved policies.
// Boards from LOST games are just as valuable as boards from won games —
// both have π* targets that are better than what the net predicted.
// Contrast with NFSP-style setups, where a best-response policy is trained
// with RL and then distilled into an average policy by supervised learning.

function Act3b_NotImitation() {
  const { localTime: lt } = useSprite();

  // Phased timing:
  // 0.0 - 0.6  title fades in
  // 0.6 - 1.5  subtitle
  // 1.5 - 3.5  WRONG column reveals (winner circled, loser discarded)
  // 3.5 - 4.5  strike-through the wrong column
  // 4.5 - 7.0  RIGHT column reveals (both games feed π* targets)

  const wrongIn = clamp((lt - 1.3) / 0.8, 0, 1);
  const strike = clamp((lt - 3.5) / 0.7, 0, 1);
  const rightIn = clamp((lt - 4.5) / 0.9, 0, 1);

  // 4 mini boards representing a self-play game.
  const gameA = [ // Agent wins
    [{x:1,y:1,c:'B'}],
    [{x:1,y:1,c:'B'},{x:0,y:2,c:'W'}],
    [{x:1,y:1,c:'B'},{x:0,y:2,c:'W'},{x:2,y:0,c:'B'}],
    [{x:1,y:1,c:'B'},{x:0,y:2,c:'W'},{x:2,y:0,c:'B'},{x:0,y:0,c:'W'}],
  ];
  const gameB = [ // Agent loses
    [{x:1,y:1,c:'B'}],
    [{x:1,y:1,c:'B'},{x:2,y:2,c:'W'}],
    [{x:1,y:1,c:'B'},{x:2,y:2,c:'W'},{x:0,y:1,c:'B'}],
    [{x:1,y:1,c:'B'},{x:2,y:2,c:'W'},{x:0,y:1,c:'B'},{x:1,y:0,c:'W'}],
  ];

  // Mini board stack (3 boards overlapping to suggest a sequence)
  const BoardStack = ({ x, y, games, outcome, outcomeColor,
                       treatment /* 'use' | 'discard' | 'use-all' */,
                       opacity = 1 }) => {
    const boardSize = 52;
    const offset = 14;
    return (
      <g style={{ opacity }}>
        {/* Game label */}
        <text x={x} y={y - 18} fontFamily="var(--mono)" fontSize={10.5}
              fill="var(--ink-soft)" letterSpacing="0.05em">
          {outcome}
        </text>
        {/* Stack of boards */}
        <foreignObject x={x} y={y} width={boardSize + offset * 3 + 10} height={boardSize + 8}>
          <div xmlns="http://www.w3.org/1999/xhtml"
               style={{ position: 'relative', height: boardSize }}>
            {games.slice(0, 4).map((stones, i) => (
              <div key={i} style={{
                position: 'absolute',
                left: i * offset, top: i * 2,
                filter: treatment === 'discard' ? 'grayscale(1) opacity(0.35)' : 'none',
                transition: 'filter 400ms',
              }}>
                <GoBoard n={3} size={boardSize} stones={stones} />
              </div>
            ))}
          </div>
        </foreignObject>
        {/* Outcome badge */}
        <g transform={`translate(${x + boardSize + offset * 3 + 16}, ${y + boardSize / 2})`}>
          <circle r={12} fill={outcomeColor} opacity={0.9} />
          <text y={3} textAnchor="middle" fontFamily="var(--mono)"
                fontSize={11} fill="var(--bg)" fontWeight={600}>
            {outcome.includes('WON') ? 'W' : 'L'}
          </text>
        </g>
      </g>
    );
  };

  return (
    <>
      {/* Header */}
      <div style={{ position: 'absolute', left: 200, top: 70, opacity: clamp(lt / 0.5, 0, 1) }}>
        <SectionLabel num="11" title="We're not imitating the winner" />
      </div>

      <div style={{ position: 'absolute', left: 200, top: 142, maxWidth: 1120,
                    opacity: clamp((lt - 0.4) / 0.7, 0, 1) }}>
        <div style={{ fontFamily: 'var(--serif)', fontSize: 17, lineHeight: 1.45,
                      color: 'var(--ink)', fontWeight: 400 }}>
          Early on I made a mistake thinking that the algorithm was to merely distill the game winner's actions back into the policy network, while masking the losing agent's actions. The goal of training isn't to copy winning trajectories. It's to distill the{' '}
          <em>MCTS-improved policy</em> π* into the network — and MCTS improves the
          policy <em>regardless of who wins the game</em>.
        </div>
      </div>

      {/* Two columns */}
      <div style={{ position: 'absolute', left: 200, top: 250,
                    display: 'flex', gap: 48, alignItems: 'flex-start' }}>

        {/* ─── WRONG column ─── */}
        <div style={{
          width: 470,
          opacity: wrongIn,
          position: 'relative',
        }}>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 10.5,
            color: '#b23a2a', letterSpacing: '0.12em',
            textTransform: 'uppercase', marginBottom: 6,
          }}>
            ✗ Imitate the winner
          </div>
          <div style={{
            fontFamily: 'var(--serif)', fontSize: 15, lineHeight: 1.45,
            color: 'var(--ink)', marginBottom: 26,
          }}>
            "The winner played good moves. Predict those, discard the loser's."
          </div>

          <svg width={470} height={180} style={{ display: 'block' }}>
            <BoardStack x={0} y={20}
                        games={gameA}
                        outcome="GAME A — AGENT WON"
                        outcomeColor="#2d7a52"
                        treatment="use" />
            <BoardStack x={0} y={120}
                        games={gameB}
                        outcome="GAME B — AGENT LOST"
                        outcomeColor="#6a5d4a"
                        treatment="discard" />

            {/* Arrows into a bin */}
            <g opacity={clamp((wrongIn - 0.3) / 0.7, 0, 1)}>
              <text x={330} y={50} fontFamily="var(--mono)" fontSize={10}
                    fill="var(--ink-soft)">→ train</text>
              <text x={330} y={148} fontFamily="var(--mono)" fontSize={10}
                    fill="var(--ink-soft)" opacity={0.55}>→ 🗑 discard</text>
            </g>
          </svg>

          {/* Strike-through overlay — sized to cover just the column's
              content (header + subtitle + svg), so it doesn't leak into
              the NFSP footnote below. */}
          {strike > 0 && (
            <svg style={{
              position: 'absolute', inset: 0,
              pointerEvents: 'none',
            }} width={470} height={250}>
              <line x1={0} y1={60}
                    x2={470 * strike} y2={60 + 180 * strike}
                    stroke="#b23a2a" strokeWidth={2.5}
                    strokeLinecap="round" opacity={0.75} />
              <line x1={470 * strike} y1={60}
                    x2={0} y2={60 + 180 * strike}
                    stroke="#b23a2a" strokeWidth={2.5}
                    strokeLinecap="round" opacity={0.75} />
            </svg>
          )}
        </div>

        {/* ─── RIGHT column ─── */}
        <div style={{
          width: 540,
          opacity: rightIn,
        }}>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 10.5,
            color: 'var(--accent-mcts)', letterSpacing: '0.12em',
            textTransform: 'uppercase', marginBottom: 6,
          }}>
            ✓ What AlphaGo actually does
          </div>
          <div style={{
            fontFamily: 'var(--serif)', fontSize: 15, lineHeight: 1.45,
            color: 'var(--ink)', marginBottom: 26,
          }}>
            Every board (won <em>and</em> lost) has a π* from MCTS that's{' '}
            <em>better than the network's own prior</em>. Train on all of them.
          </div>

          <svg width={540} height={180} style={{ display: 'block' }}>
            <BoardStack x={0} y={20}
                        games={gameA}
                        outcome="GAME A — AGENT WON"
                        outcomeColor="#2d7a52"
                        treatment="use-all" />
            <BoardStack x={0} y={120}
                        games={gameB}
                        outcome="GAME B — AGENT LOST"
                        outcomeColor="#6a5d4a"
                        treatment="use-all" />

            {/* Converging arrows → π* */}
            <g opacity={clamp((rightIn - 0.3) / 0.7, 0, 1)}>
              <path d="M 305 50 Q 360 90 400 108"
                    stroke="var(--accent-mcts)" strokeWidth={1.4}
                    fill="none" strokeLinecap="round" />
              <path d="M 305 148 Q 360 128 400 112"
                    stroke="var(--accent-mcts)" strokeWidth={1.4}
                    fill="none" strokeLinecap="round" />

              {/* π* bars (target) */}
              <g transform="translate(410, 85)">
                <rect x={-6} y={-8} width={110} height={52} rx={6}
                      fill="var(--accent-mcts-bg)"
                      stroke="var(--accent-mcts)" strokeWidth={0.8} />
                <foreignObject x={0} y={-4} width={100} height={42}>
                  <div xmlns="http://www.w3.org/1999/xhtml">
                    <PolicyBars values={[0.05, 0.08, 0.62, 0.20, 0.05]}
                                width={100} height={36}
                                color="var(--accent-mcts)" />
                  </div>
                </foreignObject>
                <text x={50} y={52} textAnchor="middle"
                      fontFamily="var(--mono)" fontSize={10.5}
                      fill="var(--accent-mcts)" fontWeight={500}>
                  π*   (target)
                </text>
              </g>
            </g>
          </svg>

          <div style={{
            marginTop: 10,
            fontFamily: 'var(--serif)', fontSize: 13, fontStyle: 'italic',
            color: 'var(--ink-soft)', lineHeight: 1.45,
            maxWidth: 450,
          }}>
            The game outcome z trains the <em>value head</em> — but the policy head
            trains on π* from every position, win or lose.
          </div>
        </div>
      </div>

    </>
  );
}

function Act3c_SelfImitationVariance() {
  const { localTime: lt } = useSprite();

  const headerOp = Easing.easeOutCubic(clamp(lt / 0.5, 0, 1));
  const storyOp = Easing.easeOutCubic(clamp((lt - 0.5) / 0.6, 0, 1));
  const mathOp = Easing.easeOutCubic(clamp((lt - 1.4) / 0.7, 0, 1));
  const varianceOp = Easing.easeOutCubic(clamp((lt - 2.5) / 0.7, 0, 1));
  const takeawayOp = Easing.easeOutCubic(clamp((lt - 4.1) / 0.8, 0, 1));

  const Formula = ({ children, accent = false }) => (
    <div style={{
      fontFamily: 'var(--mono)',
      fontSize: 15,
      lineHeight: 1.65,
      color: accent ? 'var(--accent-mcts)' : 'var(--ink)',
      background: accent ? 'var(--accent-mcts-bg)' : 'rgba(31,26,20,0.04)',
      border: `1px solid ${accent ? 'rgba(176,79,50,0.24)' : 'rgba(31,26,20,0.10)'}`,
      borderRadius: 6,
      padding: '12px 14px',
      whiteSpace: 'pre-wrap',
    }}>
      {children}
    </div>
  );

  const Bullet = ({ n, children }) => (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '28px 1fr',
      gap: 10,
      alignItems: 'start',
    }}>
      <div style={{
        width: 24, height: 24, borderRadius: 12,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(31,26,20,0.08)',
        color: 'var(--ink)',
        fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 600,
      }}>
        {n}
      </div>
      <div>{children}</div>
    </div>
  );

  return (
    <>
      <div style={{ position: 'absolute', left: 200, top: 70, opacity: headerOp }}>
        <SectionLabel num="12" title="Gradient variance of self-imitation" />
      </div>

      <div style={{
        position: 'absolute', left: 200, top: 128, right: 80,
        display: 'grid',
        gridTemplateColumns: '440px 1fr',
        gap: 28,
        alignItems: 'start',
      }}>
        <div style={{
          opacity: storyOp,
          fontFamily: 'var(--serif)',
          fontSize: 15,
          lineHeight: 1.48,
          color: 'var(--ink)',
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
        }}>
          <Bullet n="1">
            Suppose the policy is evenly matched against its opponent: true win rate 50%.
          </Bullet>
          <Bullet n="2">
            It plays 100 games of T moves. In one game, by chance, it makes one good move that flips the result. Self-imitation treats the whole winning trajectory as behavior to reinforce, so that one useful decision is diluted by roughly <span style={{ fontFamily: 'var(--mono)' }}>100T - 1</span> ordinary decisions, plus whatever is already in the replay buffer.
          </Bullet>
          <Bullet n="3">
            MCTS supervision is different: every visited state receives a locally improved target policy, rather than waiting for a binary game result to explain which move mattered.
          </Bullet>
        </div>

        <div style={{
          opacity: mathOp,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}>
          <div style={{
            fontFamily: 'var(--mono)',
            fontSize: 11,
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            color: 'var(--ink-soft)',
          }}>
            Trajectory-level credit assignment
          </div>
          <Formula accent>
{`g_RL = R(τ) Σ_t ∇θ log πθ(a_t | s_t)`}
          </Formula>
          <Formula>
{`g_DAgger = Σ_t ∇θ log πθ(a*_t | s_t)`}
          </Formula>
          <div style={{
            fontFamily: 'var(--serif)',
            fontSize: 13.5,
            lineHeight: 1.45,
            color: 'var(--ink-soft)',
          }}>
            DAgger-style supervision has no sparse terminal credit assignment problem: each state is paired with a better action or target distribution.
          </div>
        </div>
      </div>

      <div style={{
        position: 'absolute', left: 200, right: 80, top: 430,
        opacity: varianceOp,
        display: 'grid',
        gridTemplateColumns: '1fr 300px',
        gap: 24,
        alignItems: 'stretch',
      }}>
        <Formula>
{`Let u_t = ∇θ log πθ(a_t | s_t).

Var[g_RL] = E[||g_RL - E[g_RL]||²]
          = E[||g_RL||²] - ||E[g_RL]||²

E[||g_RL||²] = E[R(τ)² ||Σ_t u_t||²]
             = E[R(τ)² (Σ_t ||u_t||² + 2Σ_{i<j} u_i · u_j)]`}
        </Formula>
        
      </div>

    </>
  );
}

// ── Act 3 sub-section divider: alternate RL techniques ────────────────────
function Act3_AlternateRLTitle() {
  const { localTime: lt, duration } = useSprite();
  const op = centeredFadeInOut(lt, duration);
  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexDirection: 'column', gap: 14, opacity: op,
    }}>
      <div style={{
        fontFamily: 'var(--mono)', fontSize: 12,
        color: 'var(--ink-soft)', letterSpacing: '0.22em',
        textTransform: 'uppercase',
      }}>
        Part 2 · Aside
      </div>
      <h2 style={{
        fontFamily: 'var(--serif)', fontSize: 44, fontWeight: 400,
        color: 'var(--ink)', letterSpacing: '-0.02em',
        textAlign: 'center', maxWidth: 900, lineHeight: 1.15, margin: 0,
      }}>
        Alternate RL techniques<br/>for self-play games
      </h2>
    </div>
  );
}

// ── Act 3d: MCTS ↔ NFSP duality ────────────────────────────────────────────
// A diptych showing the same student trained by two opposite-facing teachers:
// NFSP/TD propagates value backward along observed trajectories (left),
// MCTS rolls hypothetical futures forward and backs values up the tree (right).
// Both produce a Q(s,a) at the same state s for the policy to imitate.
function Act3d_MctsVsNfsp() {
  const { localTime: lt } = useSprite();

  const headerOp = Easing.easeOutCubic(clamp(lt / 0.5, 0, 1));
  const subOp    = Easing.easeOutCubic(clamp((lt - 0.4) / 0.6, 0, 1));
  const leftIn   = Easing.easeOutCubic(clamp((lt - 1.2) / 0.9, 0, 1));
  const rightIn  = Easing.easeOutCubic(clamp((lt - 2.4) / 0.9, 0, 1));
  const unifyIn  = Easing.easeOutCubic(clamp((lt - 4.0) / 0.7, 0, 1));

  // Two panels, mirrored. Common geometry tokens.
  const PanelTag = ({ children, color }) => (
    <div style={{
      fontFamily: 'var(--mono)', fontSize: 11,
      letterSpacing: '0.18em', textTransform: 'uppercase',
      color, fontWeight: 500,
    }}>{children}</div>
  );

  const PanelName = ({ children }) => (
    <div style={{
      marginTop: 4,
      fontFamily: 'var(--serif)', fontSize: 19, fontWeight: 500,
      color: 'var(--ink)', letterSpacing: '-0.01em', lineHeight: 1.15,
    }}>{children}</div>
  );

  const PanelMethod = ({ children }) => (
    <div style={{
      marginTop: 4,
      fontFamily: 'var(--serif)', fontSize: 13, fontStyle: 'italic',
      color: 'var(--ink-soft)',
    }}>{children}</div>
  );

  const Direction = ({ children, color, bg }) => (
    <div style={{
      marginTop: 10,
      display: 'inline-flex', alignItems: 'center', gap: 8,
      fontFamily: 'var(--mono)', fontSize: 10.5,
      letterSpacing: '0.14em', textTransform: 'uppercase',
      color, background: bg,
      padding: '5px 12px', borderRadius: 999,
      fontWeight: 500, whiteSpace: 'nowrap',
    }}>{children}</div>
  );

  const Formula = ({ label, color, bg, children }) => (
    <div style={{
      marginTop: 10,
      fontFamily: 'var(--mono)', fontSize: 12.5, lineHeight: 1.4,
      color: 'var(--ink)', background: bg,
      border: `1px solid ${color}`,
      padding: '8px 12px', borderRadius: 5,
      maxWidth: 360,
    }}>
      <div style={{
        fontSize: 9.5, letterSpacing: '0.14em',
        textTransform: 'uppercase', color: 'var(--ink-soft)',
        marginBottom: 4, fontWeight: 500,
      }}>{label}</div>
      {children}
    </div>
  );

  const Caption = ({ children }) => (
    <div style={{
      marginTop: 8,
      fontFamily: 'var(--serif)', fontSize: 12.5, lineHeight: 1.4,
      color: 'var(--ink-soft)', maxWidth: 360,
    }}>{children}</div>
  );

  // Diagram dims (svg viewBox space; rendered into 380x180)
  const DIAG_W = 640, DIAG_H = 280;

  return (
    <>
      {/* Header */}
      <div style={{ position: 'absolute', left: 200, top: 70, opacity: headerOp }}>
        <SectionLabel num="13" title="Same student, opposite searches" />
      </div>

      <div style={{
        position: 'absolute', left: 200, top: 122, maxWidth: 980,
        opacity: subOp,
        fontFamily: 'var(--serif)', fontSize: 15, lineHeight: 1.45,
        color: 'var(--ink)',
      }}>
        Both methods train a policy on <em>relabeled actions</em> at a state <em>s</em>.
        They differ only in where the better action comes from — the past, or an imagined future.
      </div>

      {/* Diptych */}
      <div style={{
        position: 'absolute', left: 200, top: 198, right: 80,
        display: 'grid',
        gridTemplateColumns: '1fr 1px 1fr',
        gap: 0, alignItems: 'start',
      }}>

        {/* ─── LEFT: NFSP / TD ─── */}
        <div style={{
          opacity: leftIn,
          paddingRight: 28,
          display: 'flex', flexDirection: 'column', alignItems: 'flex-end',
          textAlign: 'right',
        }}>
          <PanelTag color="var(--ink-soft)">Teacher A</PanelTag>
          <PanelName>NFSP · Best-response via TD</PanelName>
          <PanelMethod>Q-learning along observed trajectories</PanelMethod>
          <Direction color="var(--accent-net)" bg="var(--accent-net-soft)">
            <span style={{ fontSize: 13, lineHeight: 1 }}>←</span> Search backward in time
          </Direction>

          <svg width={380} height={180} viewBox={`0 0 ${DIAG_W} ${DIAG_H}`}
               preserveAspectRatio="xMaxYMid meet"
               role="img"
               aria-label="NFSP teacher: a chain of past observed states s minus three through s, with value backing up right-to-left from the observed reward via the Bellman equation."
               style={{ marginTop: 12, overflow: 'visible' }}>
            <defs>
              <marker id="nfsp-arrL" viewBox="0 0 10 10" refX="9" refY="5"
                      markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--accent-net)"/>
              </marker>
              <marker id="nfsp-arrLink" viewBox="0 0 10 10" refX="9" refY="5"
                      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-soft)"/>
              </marker>
            </defs>

            {/* timeline ground */}
            <line x1="20" y1="240" x2="620" y2="240"
                  stroke="rgba(31,26,20,0.25)" strokeWidth="1" strokeDasharray="2 6"/>
            <text x="20" y="262" fontFamily="var(--mono)" fontSize="13"
                  fill="var(--ink-soft)" letterSpacing="2">PAST</text>
            <text x="620" y="262" fontFamily="var(--mono)" fontSize="13"
                  fill="var(--ink-soft)" letterSpacing="2" textAnchor="end">NOW</text>

            {/* observed-trajectory chain (left → right): s₋₃ → s₋₂ → s₋₁ → s → r */}
            <g>
              <line x1="100" y1="130" x2="200" y2="130" stroke="var(--ink-soft)"
                    strokeWidth="1.4" markerEnd="url(#nfsp-arrLink)"/>
              <line x1="240" y1="130" x2="340" y2="130" stroke="var(--ink-soft)"
                    strokeWidth="1.4" markerEnd="url(#nfsp-arrLink)"/>
              <line x1="380" y1="130" x2="480" y2="130" stroke="var(--ink-soft)"
                    strokeWidth="1.4" markerEnd="url(#nfsp-arrLink)"/>

              <circle cx="80"  cy="130" r="22" fill="var(--bg)"
                      stroke="var(--ink)" strokeWidth="1.6"/>
              <text x="80" y="135" fontFamily="var(--serif)" fontSize="16"
                    fontStyle="italic" fill="var(--ink)" textAnchor="middle">s₋₃</text>

              <circle cx="220" cy="130" r="22" fill="var(--bg)"
                      stroke="var(--ink)" strokeWidth="1.6"/>
              <text x="220" y="135" fontFamily="var(--serif)" fontSize="16"
                    fontStyle="italic" fill="var(--ink)" textAnchor="middle">s₋₂</text>

              <circle cx="360" cy="130" r="22" fill="var(--bg)"
                      stroke="var(--ink)" strokeWidth="1.6"/>
              <text x="360" y="135" fontFamily="var(--serif)" fontSize="16"
                    fontStyle="italic" fill="var(--ink)" textAnchor="middle">s₋₁</text>

              <circle cx="500" cy="130" r="26" fill="var(--bg)"
                      stroke="var(--ink)" strokeWidth="2.6"/>
              <text x="500" y="136" fontFamily="var(--serif)" fontSize="18"
                    fontStyle="italic" fill="var(--ink)" textAnchor="middle">s</text>

              {/* reward marker */}
              <line x1="526" y1="130" x2="582" y2="130" stroke="var(--ink-soft)"
                    strokeWidth="1.4" markerEnd="url(#nfsp-arrLink)"/>
              <rect x="582" y="112" width="36" height="36" rx="3"
                    fill="var(--bg)" stroke="var(--accent-mcts)" strokeWidth="1.6"/>
              <text x="600" y="137" fontFamily="var(--mono)" fontSize="16"
                    fontWeight="500" fill="var(--accent-mcts)" textAnchor="middle">r</text>
            </g>

            {/* value backs up: right → left (above the chain) */}
            <g>
              <path d="M 600 70 C 540 40, 500 40, 500 82"
                    fill="none" stroke="var(--accent-net)" strokeWidth="2.2"
                    markerEnd="url(#nfsp-arrL)"/>
              <path d="M 500 70 C 440 40, 400 40, 360 82"
                    fill="none" stroke="var(--accent-net)" strokeWidth="2.2"
                    markerEnd="url(#nfsp-arrL)"/>
              <path d="M 360 70 C 300 40, 260 40, 220 82"
                    fill="none" stroke="var(--accent-net)" strokeWidth="2.2"
                    markerEnd="url(#nfsp-arrL)"/>
              <path d="M 220 70 C 160 40, 120 40,  80 82"
                    fill="none" stroke="var(--accent-net)" strokeWidth="2.2"
                    markerEnd="url(#nfsp-arrL)"/>

              <text x="320" y="22" fontFamily="var(--mono)" fontSize="13"
                    fill="var(--accent-net)" letterSpacing="2" textAnchor="middle">
                VALUE BACKS UP THROUGH BELLMAN
              </text>
            </g>

            <text x="500" y="186" fontFamily="var(--mono)" fontSize="14"
                  fontWeight="500" fill="var(--accent-net)" textAnchor="middle">
              Q(s, a)
            </text>
          </svg>

          <Formula label="Bellman target at s"
                   color="rgba(80,120,180,0.28)" bg="var(--accent-net-soft)">
            Q(s, a) ← r + γ <span style={{ color: 'var(--accent-net)', fontWeight: 500 }}>max</span><sub>a′</sub> Q(s′, a′)
          </Formula>

          <Caption>
            Teacher = <em>greedy a</em>. Better actions discovered <em>in retrospect through a neural network or a Q table</em>.
          </Caption>
        </div>

        {/* center axis */}
        <div style={{
          width: 1, alignSelf: 'stretch',
          background: 'rgba(31,26,20,0.18)',
        }}/>

        {/* ─── RIGHT: MCTS ─── */}
        <div style={{
          opacity: rightIn,
          paddingLeft: 28,
          display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
        }}>
          <PanelTag color="var(--ink-soft)">Teacher B</PanelTag>
          <PanelName>MCTS · Search via simulated rollouts</PanelName>
          <PanelMethod>UCT expansion through a learned model</PanelMethod>
          <Direction color="var(--accent-mcts)" bg="var(--accent-mcts-bg)">
            Search forward in time <span style={{ fontSize: 13, lineHeight: 1 }}>→</span>
          </Direction>

          <svg width={380} height={180} viewBox={`0 0 ${DIAG_W} ${DIAG_H}`}
               preserveAspectRatio="xMinYMid meet"
               role="img"
               aria-label="MCTS teacher: from root state s, an expanding three-deep tree of imagined future states with leaf values backing up to the root."
               style={{ marginTop: 12, overflow: 'visible' }}>
            <defs>
              <marker id="mcts-arrR" viewBox="0 0 10 10" refX="9" refY="5"
                      markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--accent-mcts)"/>
              </marker>
              <marker id="mcts-arrTree" viewBox="0 0 10 10" refX="9" refY="5"
                      markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-soft)"/>
              </marker>
            </defs>

            {/* timeline ground */}
            <line x1="20" y1="240" x2="620" y2="240"
                  stroke="rgba(31,26,20,0.25)" strokeWidth="1" strokeDasharray="2 6"/>
            <text x="20"  y="262" fontFamily="var(--mono)" fontSize="13"
                  fill="var(--ink-soft)" letterSpacing="2">NOW</text>
            <text x="620" y="262" fontFamily="var(--mono)" fontSize="13"
                  fill="var(--ink-soft)" letterSpacing="2" textAnchor="end">IMAGINED FUTURES</text>

            {/* root s on left, tree expanding right */}
            <circle cx="80" cy="130" r="26" fill="var(--bg)"
                    stroke="var(--ink)" strokeWidth="2.6"/>
            <text x="80" y="136" fontFamily="var(--serif)" fontSize="18"
                  fontStyle="italic" fill="var(--ink)" textAnchor="middle">s</text>

            {/* depth 1 */}
            <g stroke="var(--ink-soft)" strokeWidth="1.4" fill="none">
              <line x1="106" y1="130" x2="194" y2="50"  markerEnd="url(#mcts-arrTree)"/>
              <line x1="106" y1="130" x2="194" y2="130" markerEnd="url(#mcts-arrTree)"/>
              <line x1="106" y1="130" x2="194" y2="210" markerEnd="url(#mcts-arrTree)"/>
            </g>
            <g>
              <circle cx="220" cy="50"  r="14" fill="var(--bg)"
                      stroke="var(--ink)" strokeWidth="1.4"/>
              <text x="220" y="54" fontFamily="var(--serif)" fontSize="12"
                    fontStyle="italic" fill="var(--ink)" textAnchor="middle">s₁</text>
              <circle cx="220" cy="130" r="14" fill="var(--bg)"
                      stroke="var(--ink)" strokeWidth="1.4"/>
              <text x="220" y="134" fontFamily="var(--serif)" fontSize="12"
                    fontStyle="italic" fill="var(--ink)" textAnchor="middle">s₂</text>
              <circle cx="220" cy="210" r="14" fill="var(--bg)"
                      stroke="var(--ink)" strokeWidth="1.4"/>
              <text x="220" y="214" fontFamily="var(--serif)" fontSize="12"
                    fontStyle="italic" fill="var(--ink)" textAnchor="middle">s₃</text>
            </g>

            {/* depth 2 */}
            <g stroke="var(--ink-soft)" strokeWidth="1.2" fill="none">
              <line x1="234" y1="50"  x2="324" y2="20"  markerEnd="url(#mcts-arrTree)"/>
              <line x1="234" y1="50"  x2="324" y2="80"  markerEnd="url(#mcts-arrTree)"/>
              <line x1="234" y1="130" x2="324" y2="110" markerEnd="url(#mcts-arrTree)"/>
              <line x1="234" y1="130" x2="324" y2="160" markerEnd="url(#mcts-arrTree)"/>
              <line x1="234" y1="210" x2="324" y2="184" markerEnd="url(#mcts-arrTree)"/>
              <line x1="234" y1="210" x2="324" y2="248" markerEnd="url(#mcts-arrTree)"/>
            </g>
            <g fill="var(--bg)" stroke="var(--ink)" strokeWidth="1.1">
              <circle cx="345" cy="20"  r="9"/>
              <circle cx="345" cy="80"  r="9"/>
              <circle cx="345" cy="110" r="9"/>
              <circle cx="345" cy="160" r="9"/>
              <circle cx="345" cy="184" r="9"/>
              <circle cx="345" cy="248" r="9"/>
            </g>

            {/* depth 3: leaves with reward markers */}
            <g stroke="var(--ink-soft)" strokeWidth="1" fill="none" opacity="0.85">
              <line x1="354" y1="20"  x2="430" y2="14"/>
              <line x1="354" y1="20"  x2="430" y2="38"/>
              <line x1="354" y1="80"  x2="430" y2="74"/>
              <line x1="354" y1="80"  x2="430" y2="98"/>
              <line x1="354" y1="110" x2="430" y2="116"/>
              <line x1="354" y1="160" x2="430" y2="150"/>
              <line x1="354" y1="160" x2="430" y2="172"/>
              <line x1="354" y1="184" x2="430" y2="194"/>
              <line x1="354" y1="248" x2="430" y2="226"/>
              <line x1="354" y1="248" x2="430" y2="252"/>
            </g>
            <g fill="var(--accent-mcts)" opacity="0.85">
              <rect x="436" y="8"   width="12" height="12" rx="2"/>
              <rect x="436" y="32"  width="12" height="12" rx="2"/>
              <rect x="436" y="68"  width="12" height="12" rx="2"/>
              <rect x="436" y="92"  width="12" height="12" rx="2"/>
              <rect x="436" y="110" width="12" height="12" rx="2"/>
              <rect x="436" y="144" width="12" height="12" rx="2"/>
              <rect x="436" y="166" width="12" height="12" rx="2"/>
              <rect x="436" y="188" width="12" height="12" rx="2"/>
              <rect x="436" y="220" width="12" height="12" rx="2"/>
              <rect x="436" y="246" width="12" height="12" rx="2"/>
            </g>
            <text x="494" y="138" fontFamily="var(--mono)" fontSize="12"
                  fill="var(--accent-mcts)" letterSpacing="2" fontWeight="500">LEAF</text>
            <text x="494" y="154" fontFamily="var(--mono)" fontSize="12"
                  fill="var(--accent-mcts)" letterSpacing="2" fontWeight="500">VALUES</text>

            {/* leaves → root, in accent */}
            <g fill="none" stroke="var(--accent-mcts)" strokeWidth="2.2">
              <path d="M 430 14  C 300 -10,  180 -10,   80 100" markerEnd="url(#mcts-arrR)"/>
              <path d="M 430 130 C 300 70,   180 70,   104 118" markerEnd="url(#mcts-arrR)"/>
              <path d="M 430 252 C 300 270,  180 270,   80 160" markerEnd="url(#mcts-arrR)"/>
            </g>

            <text x="80" y="186" fontFamily="var(--mono)" fontSize="14"
                  fontWeight="500" fill="var(--accent-mcts)" textAnchor="middle">
              Q(s, a)
            </text>
          </svg>

          <Formula label="Backed-up value at s"
                   color="rgba(176,79,50,0.24)" bg="var(--accent-mcts-bg)">
            Q(s, a) ← <span style={{ color: 'var(--accent-mcts)', fontWeight: 500 }}>𝔼</span><sub>τ ~ tree</sub> [ Σ γ<sup>t</sup> r<sub>t</sub> ]
          </Formula>

          <Caption>
            Teacher = <em>visit counts</em>. Better actions discovered <em>in foresight</em>.
          </Caption>
        </div>
      </div>

      {/* Unifying takeaway */}
      <div style={{
        position: 'absolute', left: 200, right: 80, bottom: 46,
        opacity: unifyIn,
        borderTop: '1px solid rgba(31,26,20,0.14)',
        paddingTop: 14,
        display: 'flex', gap: 18, alignItems: 'baseline',
      }}>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 10.5,
          color: 'var(--ink-soft)', letterSpacing: '0.14em',
          textTransform: 'uppercase', flexShrink: 0,
        }}>
          The duality
        </div>
        <div style={{
          fontFamily: 'var(--serif)', fontSize: 14.5, lineHeight: 1.45,
          color: 'var(--ink)', maxWidth: 900,
        }}>
          Both teachers find a better action at <em>s</em> for the student <span style={{ fontFamily: 'var(--mono)', color: 'var(--ink)' }}>π<sub>θ</sub></span> to imitate — NFSP from trajectories that <span style={{ color: 'var(--accent-net)', fontWeight: 500 }}>already happened</span>, MCTS from trajectories that <span style={{ color: 'var(--accent-mcts)', fontWeight: 500 }}>haven't</span>. The student doesn't know the difference.
        </div>
      </div>
    </>
  );
}

Object.assign(window, {
  Act3b_NotImitation,
  Act3c_SelfImitationVariance,
  Act3d_MctsVsNfsp,
  Act3_AlternateRLTitle,
});
