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
      <div style={{fontFamily:'var(--serif)', fontSize:72, fontWeight:400, color:'var(--ink)', letterSpacing:'-0.03em'}}>Implementing AlphaGo</div>
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
      <div style={{fontFamily:'var(--serif)', fontSize:72, fontWeight:400, color:'var(--ink)', letterSpacing:'-0.03em'}}>AutoGo</div>
      <div style={{fontFamily:'var(--serif)', fontSize:20, color:'var(--ink-soft)', maxWidth:720, textAlign:'center', lineHeight:1.4, marginTop:2}}>
        Implementing AlphaGo Zero from Scratch
      </div>
    </div>
  );
}

// Compute table showing how many GPU/TPU-hours each project used.
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
  const maxHours = Math.max(...rows.map(r => r.hours));
  const panelOp = Easing.easeOutCubic(clamp((lt - 0.4) / 0.5, 0, 1));

  const fmt = (n) => n.toLocaleString();

  return (
    <>
      <SlideHeader num="01" title="Motivation: what does it take to create a strong Go AI in 2026?" maxWidth={760} subtitle={<>
        AlphaGo (2016) took considerable computing resources to train. In 2020, KataGo produced a strong open-source Go AI with far fewer computational resources, and developed algorithmic techniques to speed up convergence.
        In 2026, do we still need all these tricks, or does increased GPU performance + LLM-assisted coding allow us to radically simplify the algorithmic recipe?
      </>} />

      <div style={{
        position:'absolute', left:200, top:240, right:80,
        opacity: panelOp,
      }}>
        <div style={{
          display:'grid',
          gridTemplateColumns: '200px 70px 1fr 130px 60px',
          alignItems:'center',
          columnGap: 18,
          rowGap: 0,
          fontFamily:'var(--mono)', fontSize: 11,
          color:'var(--ink-soft)', letterSpacing:'0.1em', textTransform:'uppercase',
          paddingBottom: 8,
          borderBottom: '1px solid rgba(31,26,20,0.15)',
          marginBottom: 4,
        }}>
          <div>Project</div>
          <div>Year</div>
          <div>GPU / TPU-hours</div>
          <div style={{textAlign:'right'}}>Total</div>
          <div></div>
        </div>

        {rows.map((r, i) => {
          const start = 0.8 + i * 0.6;
          const rowOp = Easing.easeOutCubic(clamp((lt - start) / 0.4, 0, 1));
          const barP  = Easing.easeOutCubic(clamp((lt - (start + 0.1)) / 1.0, 0, 1));
          const barW  = barP * (r.hours / maxHours);
          const countN = Math.round(r.hours * Easing.easeOutCubic(clamp((lt - (start + 0.1)) / 1.0, 0, 1)));
          const strong = r.name === 'AutoGo';

          return (
            <div key={r.name} style={{
              display:'grid',
              gridTemplateColumns: '200px 70px 1fr 130px 60px',
              alignItems:'center',
              columnGap: 18,
              padding:'16px 0',
              borderBottom: i < rows.length - 1 ? '1px solid rgba(31,26,20,0.08)' : 'none',
              opacity: rowOp,
              transform: `translateY(${(1 - rowOp) * 6}px)`,
            }}>
              <div style={{
                fontFamily:'var(--serif)', fontSize:18,
                color:'var(--ink)', fontWeight: strong ? 600 : 400,
              }}>
                {r.name}
                {r.tabulaRasa && <span style={{ color:'var(--ink-soft)' }}>*</span>}
                {r.preliminary && <span style={{ color:'var(--ink-soft)' }}>†</span>}
              </div>
              <div style={{
                fontFamily:'var(--mono)', fontSize:12,
                color:'var(--ink-soft)', fontVariantNumeric:'tabular-nums',
              }}>
                {r.year}
              </div>
              <div style={{ position:'relative', height:22 }}>
                <div style={{
                  position:'absolute', left:0, top:'50%', transform:'translateY(-50%)',
                  height: 14, width: `${barW * 100}%`,
                  background: strong ? 'var(--accent-mcts)' : 'var(--ink)',
                  opacity: strong ? 1 : 0.78,
                  borderRadius: 3,
                  transition: 'none',
                }} />
              </div>
              <div style={{
                textAlign:'right',
                fontFamily:'var(--mono)', fontSize:16,
                color:'var(--ink)', fontWeight: strong ? 600 : 400,
                fontVariantNumeric:'tabular-nums',
              }}>
                {fmt(countN)}
              </div>
              <div style={{ fontFamily:'var(--mono)', fontSize:10.5, position:'relative', zIndex: 5 }}>
                {r.cite && (
                  <a href={r.cite} target="_blank" rel="noopener noreferrer"
                     onClick={(e) => e.stopPropagation()}
                     style={{
                       display:'inline-block',
                       color:'var(--ink-soft)',
                       textDecoration:'underline',
                       textUnderlineOffset: 2,
                       cursor: 'pointer',
                       padding: '6px 8px',
                       margin: '-6px -8px',
                       pointerEvents: 'auto',
                     }}>
                    [ref]
                  </a>
                )}
              </div>
            </div>
          );
        })}

        <div style={{
          marginTop: 18,
          fontFamily:'var(--mono)', fontSize: 11,
          color:'var(--ink-soft)', lineHeight: 1.6,
        }}>
          <div>* trained tabula rasa.</div>
          <div>† results are preliminary; we are re-running experiments to validate.</div>
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
    { label: 'AutoGo as Black', wins: 42, total: 55, pct: 76.4,
      accent: 'var(--ink)',         op: blackOp, p: blackP, side: 'black' },
    { label: 'AutoGo as White', wins: 38, total: 49, pct: 77.6,
      accent: 'var(--accent-mcts)', op: whiteOp, p: whiteP, side: 'white' },
  ];

  return (
    <>
      <div style={{ opacity: headerOp }}>
        <SlideHeader num="12" title="AutoGo wins ~77% of games against KataGo" maxWidth={900} subtitle={<>
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
                {animated.toFixed(1)}<span style={{ color: s.accent, fontSize: 56 }}>%</span>
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
      <div style={{
        fontFamily:'var(--serif)', fontSize:72, fontWeight:400, color:'var(--ink)',
        letterSpacing:'-0.03em', lineHeight:1.12, textAlign:'center',
      }}>Research Findings</div>
      <div style={{fontFamily:'var(--serif)', fontSize:18, color:'var(--ink-soft)', maxWidth:640, textAlign:'center', lineHeight:1.55, marginTop:6}}>
        Lessons learned from implementing this from scratch
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
      <div style={{
        fontFamily:'var(--serif)', fontSize:60, fontWeight:400, color:'var(--ink)',
        letterSpacing:'-0.025em', lineHeight:1.08, textAlign:'center',
      }}>
        Automating Go Research<br/>
        with AutoGo
      </div>
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
  P2_Title,
  P4_Compute,
  P4_FindingsTitle,
  P4_WinRateVsKataGo,
});
