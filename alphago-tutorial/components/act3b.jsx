// components/act3b.jsx
// Act 3b — Misconception: we're not imitating the winner.
//
// Big idea: in self-play, MCTS search is what produces improved policies.
// Boards from LOST games are just as valuable as boards from won games —
// both have π* targets that are better than what the net predicted.
// Contrast with NFSP (AlphaStar), where no cheap lookahead exists and the
// training signal reduces to "imitate the winner."

function Act3b_NotImitation() {
  const { localTime: lt } = useSprite();

  // Phased timing (9s total):
  // 0.0 - 0.6  title fades in
  // 0.6 - 1.5  subtitle
  // 1.5 - 3.5  WRONG column reveals (winner circled, loser discarded)
  // 3.5 - 4.5  strike-through the wrong column
  // 4.5 - 7.0  RIGHT column reveals (both games feed π* targets)
  // 7.0 - 9.0  NFSP footnote

  const wrongIn = clamp((lt - 1.3) / 0.8, 0, 1);
  const strike = clamp((lt - 3.5) / 0.7, 0, 1);
  const rightIn = clamp((lt - 4.5) / 0.9, 0, 1);
  const footIn = clamp((lt - 6.8) / 0.7, 0, 1);

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
        <SectionLabel num="06" title="We're not imitating the winner" />
      </div>

      <div style={{ position: 'absolute', left: 200, top: 142, maxWidth: 1120,
                    opacity: clamp((lt - 0.4) / 0.7, 0, 1) }}>
        <div style={{ fontFamily: 'var(--serif)', fontSize: 17, lineHeight: 1.45,
                      color: 'var(--ink)', fontWeight: 400 }}>
          The goal of training isn't to copy winning trajectories. It's to distill the{' '}
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
            ✗ Neural Fictitious Self-Play
          </div>
          <div style={{
            fontFamily: 'var(--serif)', fontSize: 15, lineHeight: 1.45,
            color: 'var(--ink)', marginBottom: 26,
          }}>
            "The winner played good moves. Keep those, discard the loser's."
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

      {/* NFSP footnote */}
      {footIn > 0 && (
        <div style={{
          position: 'absolute', left: 200, bottom: 70, right: 80,
          opacity: footIn,
          padding: '12px 16px 14px',
          borderTop: '1px solid rgba(31,26,20,0.12)',
          display: 'flex', gap: 24, alignItems: 'baseline',
        }}>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 10.5,
            color: 'var(--ink-soft)', letterSpacing: '0.1em',
            textTransform: 'uppercase', flexShrink: 0,
          }}>
            aside — NFSP
          </div>
          <div style={{
            fontFamily: 'var(--serif)', fontSize: 13.5, lineHeight: 1.55,
            color: 'var(--ink)', maxWidth: 900,
          }}>
            The "reinforce the winner" approach is called{' '}
            <strong>neural fictitious self-play</strong> — it's what you fall back on when
            lookahead search isn't tractable. It powers agents like{' '}
            <strong>AlphaStar</strong> in StarCraft II, where the branching factor and
            real-time constraints rule out MCTS.
          </div>
        </div>
      )}
    </>
  );
}

Object.assign(window, { Act3b_NotImitation });
