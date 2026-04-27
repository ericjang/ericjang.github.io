// Act 7: Speedup stack — cumulative throughput as four vertical bars.

function Act7_Speedup() {
  const { localTime: lt } = useSprite();

  const bars = [
    { name: 'Python',        sub: 'single-thread',     visits:   48, mult:  1.0 },
    { name: 'C++',           sub: '+ python callback for inference',    visits:  124, mult:  2.6 },
    { name: 'game-parallel', sub: 'batched inference across 8x games',       visits:  303, mult:  6.3 },
    { name: 'leaf-parallel', sub: 'batched inference across 8x games, x8 leaves per game tree',  visits:  933, mult: 19.4 },
  ];
  const TOTAL_MULT = bars[bars.length - 1].mult;
  const AXIS_MAX   = 20;

  // Plot geometry — pushed down so the y-axis labels don't collide with
  // the (now multi-line) subtitle text.
  const PLOT_LEFT   = 240;
  const PLOT_RIGHT  = 1200;
  const PLOT_TOP    = 290;
  const PLOT_BOTTOM = 510;
  const PLOT_H      = PLOT_BOTTOM - PLOT_TOP;
  const BAR_W       = 110;
  const N           = bars.length;
  const slot        = (PLOT_RIGHT - PLOT_LEFT) / N;
  const xCenter     = (i) => PLOT_LEFT + slot * (i + 0.5);
  const xBar        = (i) => xCenter(i) - BAR_W / 2;
  const yForMult    = (m) => PLOT_BOTTOM - (Math.max(0, m) / AXIS_MAX) * PLOT_H;

  // Reveal timing
  const BAR_START = [1.0, 1.9, 2.8, 3.9];
  const BAR_LEN   = 0.9;
  const titleOp   = clamp(lt / 0.5, 0, 1);
  const subOp     = clamp((lt - 0.4) / 0.6, 0, 1);
  const axisOp    = clamp((lt - 0.7) / 0.6, 0, 1);

  const gridMarks = [0, 5, 10, 15, 20];

  return (
    <>
      <div style={{ position: 'absolute', left: 200, top: 80, opacity: titleOp }}>
        <SectionLabel num="10" title="Improving visits/sec throughput" />
      </div>

      <div style={{ position: 'absolute', left: 200, top: 150, maxWidth: 800, opacity: subOp }}>
        <Caption size={13.5} style={{ lineHeight: 1.55 }}>
          In RL, learning speed is often a function of how many steps/second you can collect environment interactions. Go is already an easy game to simulate to begin with, but we can further speed things up by implementing MCTS in C++ (avoids GIL bottlenecks when multi-threading), batching NN inference between parallel games, and multi-threading multiple MCTS simulations in parallel.
        </Caption>
      </div>

      {/* SVG: gridlines, bars, multiplier above each bar */}
      <svg style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} width="100%" height="100%">
        {/* Gridlines + y-axis labels */}
        <g opacity={axisOp}>
          {gridMarks.map(g => {
            const y = yForMult(g);
            const isBase = g === 0;
            return (
              <g key={g}>
                <line
                  x1={PLOT_LEFT - 6} y1={y}
                  x2={PLOT_RIGHT} y2={y}
                  stroke="var(--ink-soft)" strokeWidth={isBase ? 0.8 : 0.5}
                  opacity={isBase ? 0.5 : 0.18}
                  strokeDasharray={isBase ? '' : '2 4'}
                />
                <text
                  x={PLOT_LEFT - 12} y={y + 3.5}
                  textAnchor="end"
                  fontFamily="var(--mono)" fontSize="10.5"
                  fill="var(--ink-soft)"
                  fontVariantNumeric="tabular-nums"
                >
                  {g}×
                </text>
              </g>
            );
          })}
        </g>

        {/* Bars + multiplier label */}
        {bars.map((b, i) => {
          const start = BAR_START[i];
          const p = clamp((lt - start) / BAR_LEN, 0, 1);
          const grow = Easing.easeOutCubic(p);
          const fullH = (b.mult / AXIS_MAX) * PLOT_H;
          const h = Math.max(p > 0 ? 3 : 0, fullH * grow);
          const x = xBar(i);
          const yTop = PLOT_BOTTOM - h;
          const cx = x + BAR_W / 2;

          const fill = i === 3 ? 'var(--accent-mcts)'
                     : i === 2 ? 'var(--accent-mcts)'
                     : i === 1 ? 'var(--ink)'
                     :           'var(--ink-soft)';
          const fillOp = i === 0 ? 0.6 : i === 2 ? 0.6 : 1;
          const labelOp = clamp((lt - start - 0.05) / 0.4, 0, 1);

          return (
            <g key={b.name}>
              {p > 0 && (
                <rect
                  x={x} y={yTop}
                  width={BAR_W} height={h}
                  fill={fill}
                  opacity={fillOp}
                  rx={3}
                />
              )}

              <text
                x={cx} y={yTop - 10}
                textAnchor="middle"
                fontFamily="var(--mono)" fontSize="14" fontWeight="600"
                fill="var(--ink)"
                opacity={labelOp}
                fontVariantNumeric="tabular-nums"
              >
                {b.mult.toFixed(1)}×
              </text>
            </g>
          );
        })}

      </svg>

      {/* Bar captions — HTML overlay so the long sub-strings can wrap. */}
      {bars.map((b, i) => {
        const start = BAR_START[i];
        const labelOp = clamp((lt - start - 0.05) / 0.4, 0, 1);
        const labelW = slot - 16; // slot minus a small gutter on each side
        const cx = xCenter(i);
        return (
          <div key={`lab-${i}`} style={{
            position: 'absolute',
            left: cx - labelW / 2,
            top: PLOT_BOTTOM + 16,
            width: labelW,
            textAlign: 'center',
            opacity: labelOp,
            transform: `translateY(${(1 - labelOp) * 4}px)`,
          }}>
            <div style={{
              fontFamily: 'var(--mono)', fontSize: 13, fontWeight: 500,
              color: 'var(--ink)',
            }}>
              {b.name}
            </div>
            <div style={{
              fontFamily: 'var(--mono)', fontSize: 10.5,
              color: 'var(--ink-soft)',
              marginTop: 4, lineHeight: 1.4,
            }}>
              {b.sub}
            </div>
            <div style={{
              fontFamily: 'var(--mono)', fontSize: 11.5,
              color: 'var(--ink-soft)',
              marginTop: 8,
              fontVariantNumeric: 'tabular-nums',
            }}>
              {b.visits.toLocaleString()} visits/s
            </div>
          </div>
        );
      })}
    </>
  );
}

Object.assign(window, { Act7_Speedup });
