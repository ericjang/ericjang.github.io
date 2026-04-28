// components/parts.jsx
// Section title cards and cross-cutting scenes for Parts 2, 3 (AutoGo),
// and 4 (Research Findings). Plus the page-level IntroInstructions shown
// at the very start of the timeline.

// ── IntroInstructions ───────────────────────────────────────────────────────
// First slide. Shows the four arrow-key controls with a looping animation
// where each key is "pressed" in turn. The internal ticker is independent
// of the Stage timeline, so the demo plays even before any user interaction.
function IntroInstructions() {
  const { localTime: lt, duration } = useSprite();
  const slideOp = Math.min(
    Easing.easeOutCubic(clamp(lt / 0.6, 0, 1)),
    1 - Easing.easeInCubic(clamp((lt - (duration - 0.8)) / 0.8, 0, 1))
  );

  // Self-driven animation ticker so the key-press loop plays even when the
  // main timeline is paused.
  const [now, setNow] = React.useState(0);
  React.useEffect(() => {
    let raf;
    const start = performance.now();
    const tick = () => {
      setNow((performance.now() - start) / 1000);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const cycle = 5.2;
  const ct = now % cycle;
  const holdDur = 0.55;
  const pressUp    = ct >= 0.4  && ct < 0.4 + holdDur;
  const pressDown  = ct >= 1.5  && ct < 1.5 + holdDur;
  const pressLeft  = ct >= 2.6  && ct < 2.6 + holdDur;
  const pressRight = ct >= 3.7  && ct < 3.7 + holdDur;

  return (
    <div style={{
      position:'absolute', inset:0,
      display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
      gap: 24, opacity: slideOp,
    }}>
      <div style={{fontFamily:'var(--mono)', fontSize:12, color:'var(--ink-soft)', letterSpacing:'0.22em', textTransform:'uppercase'}}>
        How to navigate
      </div>
      <div style={{fontFamily:'var(--serif)', fontSize:44, fontWeight:400, color:'var(--ink)', letterSpacing:'-0.02em', textAlign:'center'}}>
        Scroll, or use the arrow keys
      </div>

      {/* Inverted-T keyboard with labels */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 110px)',
        gridTemplateRows: 'auto auto',
        justifyContent: 'center',
        alignItems: 'start',
        rowGap: 18,
        marginTop: 20,
      }}>
        <div /> {/* empty top-left */}
        <KeyCap pressed={pressUp} glyph="▲" label="play prev" />
        <div /> {/* empty top-right */}
        <KeyCap pressed={pressLeft}  glyph="◀" label="tick back" />
        <KeyCap pressed={pressDown}  glyph="▼" label="play next" />
        <KeyCap pressed={pressRight} glyph="▶" label="tick" />
      </div>

      <div style={{
        marginTop: 22,
        fontFamily:'var(--serif)', fontSize:14, color:'var(--ink-soft)',
        fontStyle:'italic', textAlign:'center', maxWidth: 560, lineHeight: 1.55,
      }}>
        The mouse wheel works too — scrolling advances the timeline smoothly.
      </div>
    </div>
  );
}

function KeyCap({ pressed, glyph, label }) {
  return (
    <div style={{
      display:'flex', flexDirection:'column', alignItems:'center', gap:8,
    }}>
      <div style={{
        width: 68, height: 68,
        display:'flex', alignItems:'center', justifyContent:'center',
        fontFamily:'var(--mono)', fontSize: 26, fontWeight: 500,
        color: pressed ? 'var(--bg)' : 'var(--ink)',
        background: pressed ? 'var(--ink)' : 'rgba(246,241,231,0.96)',
        border: '1px solid rgba(31,26,20,0.22)',
        borderRadius: 10,
        boxShadow: pressed
          ? 'inset 0 3px 6px rgba(0,0,0,0.25)'
          : '0 4px 0 rgba(31,26,20,0.15), 0 1px 2px rgba(31,26,20,0.08)',
        transform: pressed ? 'translateY(4px)' : 'translateY(0)',
        transition: 'transform 80ms ease-out, background 80ms, color 80ms, box-shadow 80ms',
      }}>
        {glyph}
      </div>
      <div style={{
        fontFamily:'var(--mono)', fontSize: 10.5,
        letterSpacing:'0.1em', textTransform:'uppercase',
        color: pressed ? 'var(--accent-mcts)' : 'var(--ink-soft)',
        fontWeight: pressed ? 600 : 400,
        transition: 'all 120ms',
      }}>
        {label}
      </div>
    </div>
  );
}

function P2_Title() {
  const { localTime: lt, duration } = useSprite();
  const op = Math.min(
    Easing.easeOutCubic(clamp(lt / 0.6, 0, 1)),
    1 - Easing.easeInCubic(clamp((lt - (duration - 0.8)) / 0.8, 0, 1))
  );
  return (
    <div style={{
      position:'absolute', inset:0, display:'flex', alignItems:'center', justifyContent:'center',
      flexDirection:'column', gap:18, opacity:op,
    }}>
      <div style={{fontFamily:'var(--mono)', fontSize:13, color:'var(--ink-soft)', letterSpacing:'0.22em', textTransform:'uppercase'}}>Part 2</div>
      <h2 style={{fontFamily:'var(--serif)', fontSize:72, fontWeight:400, color:'var(--ink)', letterSpacing:'-0.03em', margin:0}}>Implementing AlphaGo</h2>
      <div style={{fontFamily:'var(--serif)', fontSize:18, color:'var(--ink-soft)', maxWidth:640, textAlign:'center', lineHeight:1.55, marginTop:6}}>
        Now that we know how the game is played, here is how to implement a superhuman Go AI with AlphaGo.
      </div>
    </div>
  );
}

// ── Part 3 · AutoGo ─────────────────────────────────────────────────────────
function P3_AutoGoTitle() {
  const { localTime: lt, duration } = useSprite();
  const op = Math.min(
    Easing.easeOutCubic(clamp(lt / 0.6, 0, 1)),
    1 - Easing.easeInCubic(clamp((lt - (duration - 0.8)) / 0.8, 0, 1))
  );
  return (
    <div style={{
      position:'absolute', inset:0, display:'flex', alignItems:'center', justifyContent:'center',
      flexDirection:'column', gap:18, opacity:op,
    }}>
      <div style={{fontFamily:'var(--mono)', fontSize:13, color:'var(--ink-soft)', letterSpacing:'0.22em', textTransform:'uppercase'}}>Part 3</div>
      <h2 style={{fontFamily:'var(--serif)', fontSize:72, fontWeight:400, color:'var(--ink)', letterSpacing:'-0.03em', margin:0}}>AutoGo</h2>
      <div style={{fontFamily:'var(--serif)', fontSize:20, color:'var(--ink-soft)', maxWidth:720, textAlign:'center', lineHeight:1.4, marginTop:2}}>
        Implementing AlphaGo Zero from Scratch
      </div>
    </div>
  );
}

// Compute scatterplot showing how many GPU/TPU-hours each project used,
// over time, on a log y-axis. Each project is one point; AutoGo is the
// punchline so it's highlighted.
function P4_Compute() {
  const { localTime: lt } = useSprite();
  const rows = [
    { name: 'AlphaGo Lee',  year: 2016, hours:  40_000, cite: null },
    { name: 'AlphaGo Zero', year: 2017, hours: 464_000,
      cite: 'https://www.yuzeh.com/data/agz-cost.html', tabulaRasa: true },
    { name: 'KataGo',       year: 2020, hours:  12_000,
      cite: 'https://arxiv.org/pdf/1902.10565', tabulaRasa: true },
    { name: 'AutoGo',       year: 2026, hours:   3_000, cite: null,
      preliminary: true },
  ];
  const panelOp = Easing.easeOutCubic(clamp((lt - 0.4) / 0.5, 0, 1));

  // Plot geometry (in the SVG's viewBox). Margins reserve room for axis
  // labels; the data area is what's left.
  const W = 1000, H = 360;
  const M = { l: 92, r: 64, t: 28, b: 56 };
  const innerW = W - M.l - M.r;
  const innerH = H - M.t - M.b;

  const xMin = 2015, xMax = 2027;
  const yMin = 3, yMax = 6; // log10(hours): 10^3 = 1k, 10^6 = 1M
  const xScale = (yr)    => M.l + ((yr - xMin) / (xMax - xMin)) * innerW;
  const yScale = (hours) => M.t + (1 - (Math.log10(hours) - yMin) / (yMax - yMin)) * innerH;

  const xTicks = [2016, 2018, 2020, 2022, 2024, 2026];
  const yTicks = [
    { v: 1_000,    label: '1k'    },
    { v: 10_000,   label: '10k'   },
    { v: 100_000,  label: '100k'  },
    { v: 1_000_000,label: '1M'    },
  ];

  const fmt = (n) => n.toLocaleString();

  return (
    <>
      <SlideHeader num="01" title="Motivation: what does it take to create a strong Go AI in 2026?" maxWidth={760} subtitle={<>
        <div>
          AlphaGo (2016) and AlphaGo Zero (2017) took considerable computing resources to train. KataGo (2020) used various algorithmic techniques to speed up convergence, achieving ~38x reduction in compute.
          In 2026, can we train a strong AI with modest computational resources, while keeping the recipe as simple as possible?
        </div>
      </>} />

      <div style={{
        position: 'absolute', left: 200, top: 250, right: 80,
        opacity: panelOp,
      }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%"
             role="img"
             aria-label="Scatterplot of GPU/TPU-hours used to train Go AIs over time, on a log scale: AlphaGo Lee 40k hours in 2016, AlphaGo Zero 464k hours in 2017, KataGo 12k hours in 2020, AutoGo 3k hours in 2026."
             style={{ display: 'block', overflow: 'visible' }}>
          {/* Y gridlines + labels */}
          {yTicks.map((t, i) => {
            const y = yScale(t.v);
            const tickOp = Easing.easeOutCubic(clamp((lt - 0.5) / 0.4, 0, 1));
            return (
              <g key={t.v} opacity={tickOp}>
                <line x1={M.l} x2={W - M.r} y1={y} y2={y}
                      stroke="rgba(31,26,20,0.10)" strokeWidth="1"
                      strokeDasharray={i === 0 ? 'none' : '2 4'} />
                <text x={M.l - 12} y={y + 4} textAnchor="end"
                      fontFamily="var(--mono)" fontSize="13"
                      fill="var(--ink-soft)" fontVariantNumeric="tabular-nums">
                  {t.label}
                </text>
              </g>
            );
          })}

          {/* Axis spines */}
          <line x1={M.l} x2={M.l} y1={M.t} y2={H - M.b}
                stroke="rgba(31,26,20,0.35)" strokeWidth="1.2" />
          <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b}
                stroke="rgba(31,26,20,0.35)" strokeWidth="1.2" />

          {/* X tick labels */}
          {xTicks.map((yr) => {
            const x = xScale(yr);
            const tickOp = Easing.easeOutCubic(clamp((lt - 0.5) / 0.4, 0, 1));
            return (
              <g key={yr} opacity={tickOp}>
                <line x1={x} x2={x} y1={H - M.b} y2={H - M.b + 6}
                      stroke="rgba(31,26,20,0.35)" strokeWidth="1" />
                <text x={x} y={H - M.b + 22} textAnchor="middle"
                      fontFamily="var(--mono)" fontSize="13"
                      fill="var(--ink-soft)" fontVariantNumeric="tabular-nums">
                  {yr}
                </text>
              </g>
            );
          })}

          {/* Axis titles */}
          <text x={M.l - 72} y={M.t + innerH / 2}
                transform={`rotate(-90, ${M.l - 72}, ${M.t + innerH / 2})`}
                textAnchor="middle"
                fontFamily="var(--mono)" fontSize="12"
                letterSpacing="0.14em"
                fill="var(--ink-soft)"
                style={{ textTransform: 'uppercase' }}>
            GPU / TPU-hours (log)
          </text>
          <text x={M.l + innerW / 2} y={H - 8} textAnchor="middle"
                fontFamily="var(--mono)" fontSize="12"
                letterSpacing="0.14em"
                fill="var(--ink-soft)"
                style={{ textTransform: 'uppercase' }}>
            Year
          </text>

          {/* Points */}
          {rows.map((r, i) => {
            const cx = xScale(r.year);
            const cy = yScale(r.hours);
            const start = 0.9 + i * 0.35;
            const op = Easing.easeOutCubic(clamp((lt - start) / 0.45, 0, 1));
            const grow = Easing.easeOutCubic(clamp((lt - start) / 0.55, 0, 1));
            const isAuto = r.name === 'AutoGo';
            const fill = isAuto ? 'var(--accent-mcts)' : 'var(--ink)';
            const r0 = isAuto ? 9 : 7;

            // Place the project label so it doesn't crash into adjacent
            // points or the plot edge. Default = above and to the right.
            const labelDX = isAuto ? -14 : 14;
            const labelDY = isAuto ? -16 : -14;
            const labelAnchor = isAuto ? 'end' : 'start';

            return (
              <g key={r.name} opacity={op}>
                {isAuto && (
                  <circle cx={cx} cy={cy} r={r0 * 2.4 * grow}
                          fill="var(--accent-mcts)" opacity={0.12} />
                )}
                <circle cx={cx} cy={cy} r={r0 * grow}
                        fill={fill} opacity={isAuto ? 1 : 0.85} />
                <text x={cx + labelDX} y={cy + labelDY}
                      textAnchor={labelAnchor}
                      fontFamily="var(--serif)" fontSize="17"
                      fontWeight={isAuto ? 600 : 500}
                      fill="var(--ink)">
                  {r.name}
                  {r.tabulaRasa && <tspan fill="var(--ink-soft)">*</tspan>}
                  {r.preliminary && <tspan fill="var(--ink-soft)">†</tspan>}
                </text>
                <text x={cx + labelDX} y={cy + labelDY + 16}
                      textAnchor={labelAnchor}
                      fontFamily="var(--mono)" fontSize="12"
                      fontVariantNumeric="tabular-nums"
                      fill="var(--ink-soft)">
                  {fmt(r.hours)} h
                </text>
              </g>
            );
          })}
        </svg>

        {/* Reference links + footnotes underneath the plot */}
        <div style={{
          marginTop: 8,
          display: 'flex', gap: 18, flexWrap: 'wrap',
          fontFamily: 'var(--mono)', fontSize: 11,
          color: 'var(--ink-soft)', lineHeight: 1.6,
        }}>
          {rows.filter(r => r.cite).map(r => (
            <a key={r.name}
               href={r.cite} target="_blank" rel="noopener noreferrer"
               onClick={(e) => e.stopPropagation()}
               style={{
                 color: 'var(--ink-soft)',
                 textDecoration: 'underline', textUnderlineOffset: 2,
                 pointerEvents: 'auto',
               }}>
              [{r.name} ref]
            </a>
          ))}
          <span style={{ marginLeft: 'auto' }}>* trained tabula rasa. † preliminary; re-running to validate.</span>
        </div>
      </div>
    </>
  );
}

// ── Part 4 · Research Findings ──────────────────────────────────────────────

// Headline finding: AutoGo's win rate vs the KataGo reference checkpoint,
// split by which color AutoGo played. Two big stats with bars; numbers
// count up.
function P4_WinRateVsKataGo() {
  const { localTime: lt } = useSprite();

  const headerOp = Easing.easeOutCubic(clamp(lt / 0.5, 0, 1));
  const blackOp  = Easing.easeOutCubic(clamp((lt - 0.6) / 0.5, 0, 1));
  const whiteOp  = Easing.easeOutCubic(clamp((lt - 1.2) / 0.5, 0, 1));
  const blackP   = Easing.easeOutCubic(clamp((lt - 0.7) / 1.6, 0, 1));
  const whiteP   = Easing.easeOutCubic(clamp((lt - 1.3) / 1.6, 0, 1));
  const footOp   = Easing.easeOutCubic(clamp((lt - 2.4) / 0.8, 0, 1));

  const STATS = [
    { label: 'AutoGo as Black', wins: 42, total: 55, pct: 76,
      accent: 'var(--ink)',         op: blackOp, p: blackP, side: 'black' },
    { label: 'AutoGo as White', wins: 38, total: 49, pct: 77,
      accent: 'var(--ink)', op: whiteOp, p: whiteP, side: 'white' },
  ];

  return (
    <>
      <div style={{ opacity: headerOp }}>
        <SlideHeader num="14" title="AutoGo wins ~77% of games against KataGo" maxWidth={900} subtitle={<>
          On 19×19, AutoGo defeats KataGo — a strong open-source Go AI — at roughly the same rate playing as either color over 104 evaluation games.
        </>} />
      </div>

      <div style={{
        position:'absolute', left: 200, right: 80, top: 240,
        display:'flex', gap: 28,
      }}>
        {STATS.map((s) => {
          const animated = s.pct * s.p;
          const losses = s.total - s.wins;
          return (
            <div key={s.label} style={{
              flex: 1,
              padding: '32px 40px 28px',
              background: 'rgba(31,26,20,0.03)',
              border: '1px solid rgba(31,26,20,0.10)',
              borderRadius: 10,
              opacity: s.op,
              transform: `translateY(${(1 - s.op) * 8}px)`,
              display: 'flex', flexDirection: 'column', gap: 16,
            }}>
              <div style={{
                display:'flex', alignItems:'center', gap: 10,
                fontFamily: 'var(--mono)', fontSize: 11,
                letterSpacing: '0.2em', textTransform: 'uppercase',
                color: 'var(--ink-soft)',
              }}>
                <span style={{
                  display:'inline-block', width: 12, height: 12, borderRadius: '50%',
                  background: s.side === 'black' ? 'var(--stone-black)' : 'var(--stone-white)',
                  border: s.side === 'white' ? '1px solid rgba(31,26,20,0.35)' : 'none',
                }}/>
                {s.label}
              </div>

              <div style={{
                fontFamily: 'var(--serif)', fontWeight: 500,
                fontSize: 96, color: 'var(--ink)',
                fontVariantNumeric: 'tabular-nums', letterSpacing: '-0.02em',
                lineHeight: 1,
              }}>
                {Math.round(animated)}<span style={{ color: s.accent, fontSize: 56 }}>%</span>
              </div>

              <div style={{
                position:'relative',
                width: '100%', height: 10,
                background: 'rgba(31,26,20,0.08)', borderRadius: 5,
                overflow: 'hidden',
              }}>
                <div style={{
                  height: '100%', width: `${animated}%`,
                  background: s.accent, borderRadius: 5,
                }} />
              </div>

              <div style={{
                display:'flex', justifyContent:'space-between',
                fontFamily: 'var(--mono)', fontSize: 13,
                color: 'var(--ink-soft)', fontVariantNumeric: 'tabular-nums',
              }}>
                <span>
                  <span style={{color:'var(--ink)', fontWeight:600}}>{s.wins}</span>
                  &nbsp;wins · {losses} losses
                </span>
                <span>{s.wins} / {s.total}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{
        position: 'absolute', left: 200, right: 80, bottom: 56,
        fontFamily: 'var(--mono)', fontSize: 11,
        color: 'var(--ink-soft)', letterSpacing: '0.04em',
        opacity: footOp,
        lineHeight: 1.6,
      }}>
        Opponent: <span style={{color:'var(--ink)'}}>kata1-zhizi-b40c768nbt-fdx6c</span>
        &nbsp;· evaluated 2026-04-06
      </div>
    </>
  );
}

// Three-card reflection on what current LLM coding assistants can and
// can't do for this kind of from-scratch ML research project.
function P4_LLMAutoresearch() {
  const { localTime: lt } = useSprite();
  const headerOp = Easing.easeOutCubic(clamp(lt / 0.5, 0, 1));
  const card1Op  = Easing.easeOutCubic(clamp((lt - 0.5) / 0.6, 0, 1));
  const card2Op  = Easing.easeOutCubic(clamp((lt - 1.0) / 0.6, 0, 1));
  const card3Op  = Easing.easeOutCubic(clamp((lt - 1.5) / 0.6, 0, 1));
  const card4Op  = Easing.easeOutCubic(clamp((lt - 2.0) / 0.6, 0, 1));

  const Column = ({ op, accent, eyebrow, title, body, glyph, image, imageAlt, flex = '1 1 0' }) => (
    <div style={{
      flex, minWidth: 0, minHeight: 0,
      padding: '16px 22px',
      background: 'var(--bg)',
      border: '1px solid rgba(31,26,20,0.12)',
      borderTop: `4px solid ${accent}`,
      borderRadius: 6,
      opacity: op,
      transform: `translateY(${(1 - op) * 8}px)`,
      transition: 'transform 240ms ease',
      display: 'flex', flexDirection: 'column', gap: 8,
      overflow: 'hidden',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        fontFamily: 'var(--mono)', fontSize: 11,
        letterSpacing: '0.14em', textTransform: 'uppercase',
        color: accent, fontWeight: 600,
      }}>
        <span style={{
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          width: 22, height: 22, borderRadius: 11,
          border: `2px solid ${accent}`,
          fontSize: 13, fontFamily: 'var(--serif)', fontWeight: 700,
        }}>
          {glyph}
        </span>
        {eyebrow}
      </div>
      <div style={{
        fontFamily: 'var(--serif)', fontSize: 19, fontWeight: 500,
        color: 'var(--ink)', lineHeight: 1.2,
      }}>
        {title}
      </div>
      <div style={{
        fontFamily: 'var(--serif)', fontSize: 14,
        color: 'var(--ink-soft)', lineHeight: 1.45,
      }}>
        {body}
      </div>
      {image && (
        <div style={{
          flex: '1 1 0', minHeight: 0,
          background: 'rgba(31,26,20,0.04)',
          border: '1px solid rgba(31,26,20,0.10)',
          borderRadius: 4,
          padding: 6,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          overflow: 'hidden',
        }}>
          <img
            src={image}
            alt={imageAlt || ''}
            style={{
              display: 'block',
              maxWidth: '100%', maxHeight: '100%',
              objectFit: 'contain',
            }}
          />
        </div>
      )}
    </div>
  );

  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', flexDirection: 'column',
      paddingLeft: 200, paddingRight: 60,
      paddingTop: 76, paddingBottom: 56,
      gap: 20,
    }}>
      <div style={{ opacity: headerOp }}>
        <SectionLabel num="18" title="Autoresearch with LLMs: what works, what doesn't?" />
        
      </div>

      <div style={{
        flex: '1 1 auto', minHeight: 0,
        display: 'flex', gap: 20, alignItems: 'stretch',
      }}>
        <div style={{
          flex: '1 1 0', minWidth: 0,
          display: 'flex', flexDirection: 'column', gap: 20,
        }}>
          <Column
            op={card1Op}
            accent="#1f8a5b"
            glyph="✓"
            eyebrow="Works"
            title="Implementing and running experiments"
            body={<>
              Current models (Claude Opus 4.7) can implement experiments and get them to run. They can rewrite Python into C++ and do a good job at distributed systems design (protocols and gRPC services).
            </>}
            flex="0 0 auto"
          />
          <Column
            op={card2Op}
            accent="#1f8a5b"
            glyph="✓"
            eyebrow="Works"
            title="Optimizing hyperparameters"
            body={<>
              Given a "standard deep learning task" like "make the loss go down", agents can make substantial improvements by tweaking hyperparameters.
            </>}
            image="/alphago-tutorial/val_loss_progress.png"
            imageAlt="Val loss running-best curve over 63 autoresearch experiments"
            flex="1 1 0"
          />
        </div>
        <div style={{
          flex: '1 1 0', minWidth: 0,
          display: 'flex', flexDirection: 'column', gap: 20,
        }}>
          <Column
            op={card3Op}
            accent="#b64242"
            glyph="✕"
            eyebrow="Still doesn't work"
            title="Choosing the next card or switching to a new row"
            body={<>
              LLM agents cannot (yet) suggest what the next <em>card</em> in the chain should be, or recognize when we should abandon the current row and pursue a different one (lateral thinking). Strategy at the level of the lineage tree is still up to the human.
            </>}
            flex="0 0 auto"
          />
          <Column
            op={card4Op}
            accent="#b64242"
            glyph="✕"
            eyebrow="Still doesn't work"
            title="Visualization that yields an &ldquo;aha&rdquo; moment"
            body={<>
              Compacting an enormous amount of research data &amp; context into a single plot, glancing at it with computer vision, then having an &ldquo;aha&rdquo; moment of geometric understanding. For me, this moment came from a plot of MCTS argmax vs. raw policy argmax disagreement and realizing that this indicated the quality of the training signal to the policy network.
            </>}
            flex="1 1 0"
          />
        </div>
      </div>
    </div>
  );
}

// Why Go is a good outer-loop testbed for autoresearch: a box of frontier
// ML topics that all collapse into a single quick-to-verify game.
function P4_WhyGo() {
  const { localTime: lt } = useSprite();
  const headerOp = Easing.easeOutCubic(clamp(lt / 0.5, 0, 1));
  const boxOp    = Easing.easeOutCubic(clamp((lt - 0.4) / 0.5, 0, 1));
  const arrowOp  = Easing.easeOutCubic(clamp((lt - 1.4) / 0.5, 0, 1));
  const boardOp  = Easing.easeOutCubic(clamp((lt - 1.7) / 0.6, 0, 1));

  const topics = [
    'Architectures and compute multipliers',
    'Distributed RL systems',
    'Self-play, Nash equilibria, mixed strategies',
    'Recursive self-improvement',
    'synthetic reasoning data',
    'Train-time and test-time scaling',
    'Combining Onpolicy + Offpolicy RL',
    'Predicting experiment outcomes',
    'Lateral creative thinking',
    'Scientific understanding'
  ];

  // A plausible mid-game scatter of stones on a 9×9 board. Hand-picked so
  // the board reads as "a real game in progress" rather than a uniform grid.
  const stones = [
    { x: 2, y: 2, color: 'B' }, { x: 6, y: 2, color: 'W' },
    { x: 4, y: 3, color: 'B' }, { x: 3, y: 4, color: 'W' },
    { x: 5, y: 5, color: 'B' }, { x: 6, y: 6, color: 'W' },
    { x: 1, y: 5, color: 'B' }, { x: 7, y: 4, color: 'W' },
    { x: 3, y: 6, color: 'B' }, { x: 5, y: 1, color: 'W' },
    { x: 4, y: 7, color: 'B' }, { x: 1, y: 7, color: 'W' },
  ];

  // Stagger the inner topic cards' entrance.
  const cardOp = (i) => Easing.easeOutCubic(clamp((lt - 0.6 - i * 0.10) / 0.5, 0, 1));

  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', flexDirection: 'column',
      paddingLeft: 200, paddingRight: 60,
      paddingTop: 76, paddingBottom: 56,
      gap: 18,
    }}>
      <div style={{ opacity: headerOp }}>
        <SectionLabel num="20" title="Why Go for autoresearch?" />
        <div style={{
          marginTop: 12,
          fontFamily: 'var(--serif)', fontSize: 15,
          color: 'var(--ink-soft)',
          maxWidth: 920, lineHeight: 1.55,
        }}>
          Go folds a sampling platter of frontier ML problems into one outer loop. We can set LLMs about learning to do these tasks with onpolicy RL, and yet score the final result easily using Go. The ultimate RL objective is not to train strong Go agents; it is to build a playground for an automated scientist for which we have a complex-but-verifiable domain.
        </div>
      </div>

      <div style={{
        flex: '1 1 auto', minHeight: 0,
        display: 'flex', alignItems: 'stretch', gap: 28,
      }}>
        {/* Frontier-topics box */}
        <div style={{
          flex: '1 1 0', minWidth: 0, minHeight: 0,
          padding: '20px 22px',
          background: 'var(--bg)',
          border: '1px solid rgba(31,26,20,0.16)',
          borderRadius: 8,
          opacity: boxOp,
          display: 'flex', flexDirection: 'column', gap: 14,
        }}>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 11,
            letterSpacing: '0.14em', textTransform: 'uppercase',
            color: 'var(--ink-soft)', fontWeight: 600,
          }}>
            Auto-research tasks (Inner loop)
          </div>
          <div style={{
            flex: '1 1 auto', minHeight: 0,
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gridAutoRows: 'minmax(0, 1fr)',
            gap: 10,
          }}>
            {topics.map((t, i) => (
              <div key={t} style={{
                padding: '10px 14px',
                background: 'rgba(31,26,20,0.04)',
                border: '1px solid rgba(31,26,20,0.10)',
                borderLeft: '3px solid var(--accent-mcts)',
                borderRadius: 5,
                fontFamily: 'var(--serif)', fontSize: 14,
                color: 'var(--ink)', lineHeight: 1.3,
                display: 'flex', alignItems: 'center',
                opacity: cardOp(i),
                transform: `translateY(${(1 - cardOp(i)) * 6}px)`,
                transition: 'transform 240ms ease',
              }}>
                {t}
              </div>
            ))}
          </div>
        </div>

        {/* Arrow + 9×9 Go board on the right */}
        <div style={{
          flex: '0 0 auto',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <ArrowToBoard opacity={arrowOp} />
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10,
            opacity: boardOp,
          }}>
            <GoBoard n={9} size={232} padding={16} stones={stones} />
            <div style={{
              fontFamily: 'var(--mono)', fontSize: 10,
              letterSpacing: '0.16em', textTransform: 'uppercase',
              color: 'var(--accent-mcts)', fontWeight: 600,
            }}>
              Outer loop
            </div>
            <div style={{
              fontFamily: 'var(--serif)', fontSize: 13,
              color: 'var(--ink-soft)', fontStyle: 'italic',
              textAlign: 'center', maxWidth: 232, lineHeight: 1.4,
            }}>
              The output is an agent which we can score easily with a simple, quick game.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Horizontal arrow used by P4_WhyGo to point from the topics box at the
// 9×9 board. Defined out-of-line so the marker id is stable.
function ArrowToBoard({ opacity = 1 }) {
  return (
    <svg width={68} height={36} style={{ overflow: 'visible', opacity, transition: 'opacity 240ms ease' }}>
      <defs>
        <marker
          id="whygo-arrow-head"
          viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="9" markerHeight="9" orient="auto"
        >
          <path d="M 0,0 L 10,5 L 0,10 z" fill="var(--ink-soft)" />
        </marker>
      </defs>
      <line
        x1={2} y1={18} x2={56} y2={18}
        stroke="var(--ink-soft)" strokeWidth={2.4}
        strokeLinecap="round"
        markerEnd="url(#whygo-arrow-head)"
      />
    </svg>
  );
}

function P4_FindingsTitle() {
  const { localTime: lt, duration } = useSprite();
  const op = Math.min(
    Easing.easeOutCubic(clamp(lt / 0.6, 0, 1)),
    1 - Easing.easeInCubic(clamp((lt - (duration - 0.8)) / 0.8, 0, 1))
  );
  return (
    <div style={{
      position:'absolute', inset:0, display:'flex', alignItems:'center', justifyContent:'center',
      flexDirection:'column', gap:18, opacity:op,
    }}>
      <div style={{fontFamily:'var(--mono)', fontSize:13, color:'var(--ink-soft)', letterSpacing:'0.22em', textTransform:'uppercase'}}>Part 4</div>
      <h2 style={{
        fontFamily:'var(--serif)', fontSize:72, fontWeight:400, color:'var(--ink)',
        letterSpacing:'-0.03em', lineHeight:1.12, textAlign:'center', margin:0,
      }}>Research Findings</h2>
      <div style={{fontFamily:'var(--serif)', fontSize:18, color:'var(--ink-soft)', maxWidth:640, textAlign:'center', lineHeight:1.55, marginTop:6}}>
        Here are some things I've learned so far. More results coming soon
      </div>
    </div>
  );
}

// ── Tutorial cover ──────────────────────────────────────────────────────────
// The very first slide — overall framing for the whole page.
function TutorialCover() {
  const { localTime: lt, duration } = useSprite();
  const op = Math.min(
    Easing.easeOutCubic(clamp(lt / 0.6, 0, 1)),
    1 - Easing.easeInCubic(clamp((lt - (duration - 0.8)) / 0.8, 0, 1))
  );
  return (
    <div style={{
      position:'absolute', inset:0, display:'flex', alignItems:'center', justifyContent:'center',
      flexDirection:'column', gap:20, opacity:op,
    }}>
      <div style={{fontFamily:'var(--mono)', fontSize:12, color:'var(--ink-soft)', letterSpacing:'0.22em', textTransform:'uppercase'}}>
        A tutorial
      </div>
      <h1 style={{
        fontFamily:'var(--serif)', fontSize:60, fontWeight:400, color:'var(--ink)',
        letterSpacing:'-0.025em', lineHeight:1.08, textAlign:'center',
        margin: 0,
      }}>
        Automating Go Research<br/>
        with AutoGo
      </h1>
      <div style={{fontFamily:'var(--serif)', fontSize:18, color:'var(--ink-soft)', maxWidth:680, textAlign:'center', lineHeight:1.55, marginTop:4}}>
        Building a strong Go AI from scratch with modern AI tools.
      </div>

      <a
        href="https://autogo.evjang.com"
        target="_blank"
        rel="noopener noreferrer"
        onClick={(e) => e.stopPropagation()}
        style={{
          marginTop: 18,
          display: 'inline-flex', alignItems: 'center', gap: 8,
          fontFamily: 'var(--mono)', fontSize: 12,
          letterSpacing: '0.14em', textTransform: 'uppercase',
          color: 'var(--accent-mcts)',
          textDecoration: 'none',
          padding: '10px 18px',
          border: '1px solid var(--accent-mcts)',
          borderRadius: 22,
          cursor: 'pointer',
          pointerEvents: 'auto',
          position: 'relative',
          zIndex: 5,
          transition: 'background 140ms, color 140ms',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'var(--accent-mcts)';
          e.currentTarget.style.color = 'var(--bg)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'transparent';
          e.currentTarget.style.color = 'var(--accent-mcts)';
        }}
      >
        Play at autogo.evjang.com →
      </a>

      <a
        href="https://github.com/ericjang/autogo"
        target="_blank"
        rel="noopener noreferrer"
        onClick={(e) => e.stopPropagation()}
        style={{
          marginTop: 10,
          display: 'inline-flex', alignItems: 'center', gap: 8,
          fontFamily: 'var(--mono)', fontSize: 12,
          letterSpacing: '0.14em', textTransform: 'uppercase',
          color: 'var(--ink-soft)',
          textDecoration: 'none',
          padding: '10px 18px',
          border: '1px solid var(--ink-soft)',
          borderRadius: 22,
          cursor: 'pointer',
          pointerEvents: 'auto',
          position: 'relative',
          zIndex: 5,
          transition: 'background 140ms, color 140ms',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'var(--ink-soft)';
          e.currentTarget.style.color = 'var(--bg)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'transparent';
          e.currentTarget.style.color = 'var(--ink-soft)';
        }}
      >
        Code on Github →
      </a>
    </div>
  );
}

// ── AutoGo · Outro (closing CTAs) ──────────────────────────────────────────
// Echoes the cover's Play/Code buttons at the end of the deck, plus an
// "About me" link out to evjang.com.

// IMPORTANT: this pill component is defined at module scope (outside
// AutoGo_Outro). If it were defined inside the per-frame render of the
// outro, React would treat it as a new component type on every tick and
// remount the underlying <a>, which breaks clicks (mousedown lands on the
// old node, mouseup on a freshly mounted one).
function OutroPill({ href, color, label, opacity }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      onClick={(e) => e.stopPropagation()}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 8,
        fontFamily: 'var(--mono)', fontSize: 12,
        letterSpacing: '0.14em', textTransform: 'uppercase',
        color, background: 'transparent',
        textDecoration: 'none',
        padding: '10px 18px',
        border: `1px solid ${color}`,
        borderRadius: 22,
        cursor: 'pointer',
        pointerEvents: 'auto',
        position: 'relative', zIndex: 5,
        opacity,
        transform: `translateY(${(1 - opacity) * 6}px)`,
        transition: 'background 140ms, color 140ms, transform 240ms ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = color;
        e.currentTarget.style.color = 'var(--bg)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent';
        e.currentTarget.style.color = color;
      }}
    >
      {label}
    </a>
  );
}

function AutoGo_Outro() {
  const { localTime: lt, duration } = useSprite();
  const op = Math.min(
    Easing.easeOutCubic(clamp(lt / 0.6, 0, 1)),
    1 - Easing.easeInCubic(clamp((lt - (duration - 0.8)) / 0.8, 0, 1))
  );
  const headOp = Easing.easeOutCubic(clamp((lt - 0.2) / 0.6, 0, 1));
  const ctaOp  = (i) => Easing.easeOutCubic(clamp((lt - 0.6 - i * 0.18) / 0.5, 0, 1));

  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexDirection: 'column', gap: 18, opacity: op,
    }}>
      <div style={{
        fontFamily: 'var(--mono)', fontSize: 12,
        color: 'var(--ink-soft)', letterSpacing: '0.22em',
        textTransform: 'uppercase', opacity: headOp,
      }}>
        Thanks for reading
      </div>
      <div style={{
        fontFamily: 'var(--serif)', fontSize: 48, fontWeight: 400,
        color: 'var(--ink)', letterSpacing: '-0.025em',
        lineHeight: 1.1, textAlign: 'center', maxWidth: 900,
        opacity: headOp, margin: 0,
      }}>
        Play, fork, or get in touch.
      </div>

      <div style={{
        marginTop: 22,
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12,
      }}>
        <OutroPill href="https://autogo.evjang.com"
                   color="var(--accent-mcts)"
                   label="Play at autogo.evjang.com →"
                   opacity={ctaOp(0)} />
        <OutroPill href="https://github.com/ericjang/autogo"
                   color="var(--ink-soft)"
                   label="Code on Github →"
                   opacity={ctaOp(1)} />
        <OutroPill href="https://evjang.com"
                   color="var(--ink-soft)"
                   label="About me · evjang.com →"
                   opacity={ctaOp(2)} />
      </div>
    </div>
  );
}

// ── AutoGo · Thanks to Prime Intellect ─────────────────────────────────────
function AutoGo_Thanks() {
  const { localTime: lt, duration } = useSprite();
  const op = Math.min(
    Easing.easeOutCubic(clamp(lt / 0.6, 0, 1)),
    1 - Easing.easeInCubic(clamp((lt - (duration - 0.8)) / 0.8, 0, 1))
  );
  const thanksOp = Easing.easeOutCubic(clamp((lt - 0.3) / 0.6, 0, 1));
  const goOp     = Easing.easeOutCubic(clamp((lt - 2.4) / 0.8, 0, 1));

  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexDirection: 'column', gap: 36,
      opacity: op,
    }}>

      <div style={{
        fontFamily: 'var(--serif)', fontSize: 28, fontWeight: 400,
        color: 'var(--ink)', letterSpacing: '-0.01em',
        lineHeight: 1.45, textAlign: 'center', maxWidth: 860,
        opacity: thanksOp,
      }}>
        Thank you to{' '}
        <a href="https://www.primeintellect.ai/"
           target="_blank"
           rel="noopener noreferrer"
           onClick={(e) => e.stopPropagation()}
           style={{
             color: 'var(--accent-mcts)',
             textDecoration: 'underline',
             textUnderlineOffset: 4,
             cursor: 'pointer',
             display: 'inline-block',
             padding: '2px 4px',
             margin: '-2px -4px',
             pointerEvents: 'auto',
             position: 'relative',
             zIndex: 5,
           }}>
          Prime Intellect
        </a>
        {' '}for a <span style={{fontFamily:'var(--mono)', fontSize:'0.92em'}}>$10,000 USD</span> grant to enable this research project.
      </div>

      <div style={{
        fontFamily: 'var(--serif)', fontSize: 56, fontWeight: 500,
        color: 'var(--ink)', letterSpacing: '-0.025em',
        fontStyle: 'italic',
        opacity: goOp,
        transform: `translateY(${(1 - goOp) * 10}px)`,
      }}>
        Let's Go.
      </div>
    </div>
  );
}

Object.assign(window, {
  IntroInstructions,
  TutorialCover,
  AutoGo_Thanks,
  AutoGo_Outro,
  P2_Title,
  P4_Compute,
  P4_FindingsTitle,
  P4_LLMAutoresearch,
  P4_WhyGo,
  P4_WinRateVsKataGo,
});
