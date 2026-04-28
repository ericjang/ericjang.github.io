// components/part1.jsx
// Part 1: Basic Rules of Go. 10 scenes, all on a 5×5 board.

// ── Helpers ─────────────────────────────────────────────────────────────────
const BOARD_X = 200;
const BOARD_Y = 200;
const BOARD_SIZE = 360;

function fadeIn(lt, at, dur = 0.4, ease = Easing.easeOutCubic) {
  return ease(clamp((lt - at) / dur, 0, 1));
}

function centeredFadeInOut(lt, duration, enter = 0.6, exit = 0.8) {
  return Math.min(
    Easing.easeOutCubic(clamp(lt / enter, 0, 1)),
    1 - Easing.easeInCubic(clamp((lt - (duration - exit)) / exit, 0, 1))
  );
}

// ── P1_Title ────────────────────────────────────────────────────────────────
function P1_Title() {
  const { localTime: lt, duration } = useSprite();
  const op = centeredFadeInOut(lt, duration);
  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexDirection: 'column', gap: 18, opacity: op,
    }}>
      <div style={{fontFamily:'var(--mono)', fontSize:13, color:'var(--ink-soft)', letterSpacing:'0.22em', textTransform:'uppercase'}}>Part 1</div>
      <h2 style={{fontFamily:'var(--serif)', fontSize:72, fontWeight:400, color:'var(--ink)', letterSpacing:'-0.03em', margin:0}}>Rules of Go</h2>
    </div>
  );
}

// ── P1_StrategiesTitle ──────────────────────────────────────────────────────
// Sub-section divider between the rules slides and the strategy slides.
function P1_StrategiesTitle() {
  const { localTime: lt, duration } = useSprite();
  const op = centeredFadeInOut(lt, duration);
  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      flexDirection: 'column', gap: 14, opacity: op,
    }}>
      <div style={{fontFamily:'var(--mono)', fontSize:12, color:'var(--ink-soft)', letterSpacing:'0.22em', textTransform:'uppercase'}}>
        Part 1
      </div>
      <h2 style={{fontFamily:'var(--serif)', fontSize:56, fontWeight:400, color:'var(--ink)', letterSpacing:'-0.02em', margin:0}}>
        Basic Tactics in Go
      </h2>
    </div>
  );
}

// ── P1_PlacingStones ────────────────────────────────────────────────────────
// Introduce the basic mechanics: alternating turns, intersection placement,
// immovable stones, no playing on an occupied point.
function P1_PlacingStones() {
  const { localTime: lt } = useSprite();

  const moves = [
    { at: 0.5, x: 2, y: 2, color: 'B', number: 1 },
    { at: 1.6, x: 3, y: 2, color: 'W', number: 2 },
    { at: 2.5, x: 1, y: 2, color: 'B', number: 3 },
    { at: 3.4, x: 2, y: 3, color: 'W', number: 4 },
    { at: 4.3, x: 2, y: 1, color: 'B', number: 5 },
  ];

  return (
    <>
      <SlideHeader num="01" title="Placing Stones" subtitle={
        <>Black plays first. Players alternate, placing one stone per turn on any empty <em>intersection</em> — not inside the squares.</>
      } />
      <div style={{position:'absolute', left:BOARD_X, top:BOARD_Y}}>
        <GoAnim n={5} size={BOARD_SIZE} lt={lt} moves={moves} />
      </div>
      <div style={{position:'absolute', left:620, top:240, width:500, fontFamily:'var(--serif)', fontSize:15, color:'var(--ink)', lineHeight:1.65}}>
        <div style={{opacity: fadeIn(lt, 0.4)}}>
          <div style={{fontFamily:'var(--mono)', fontSize:11, color:'var(--ink-soft)', letterSpacing:'0.1em', textTransform:'uppercase', marginBottom:10}}>Rules</div>
          <ul style={{margin:0, paddingLeft:20, color:'var(--ink-soft)', lineHeight:1.75, fontSize:14}}>
            <li>Black moves first.</li>
            <li>Players alternate — one stone per turn.</li>
            <li>Stones sit on line intersections, not inside squares.</li>
            <li>Once placed, stones don't move (unless captured).</li>
          </ul>
        </div>
      </div>
    </>
  );
}

// ── P1_Scoring ──────────────────────────────────────────────────────────────
// Symmetric end position; both sides score 10, komi (7.5) tips it to white.
const SCORING_BLACK = [[1,0],[0,1],[1,1],[1,2],[0,3],[1,3],[1,4]];
const SCORING_WHITE = [[3,0],[3,1],[4,1],[3,2],[3,3],[4,3],[3,4]];
const SCORING_B_TERR = [{x:0,y:0},{x:0,y:2},{x:0,y:4}];
const SCORING_W_TERR = [{x:4,y:0},{x:4,y:2},{x:4,y:4}];
const SCORING_DAME = [{x:2,y:0},{x:2,y:1},{x:2,y:2},{x:2,y:3},{x:2,y:4}];

function P1_Scoring() {
  const { localTime: lt } = useSprite();
  const moves = [
    ...SCORING_BLACK.map(([x,y], i) => ({at: 0.05 * i,                x, y, color: 'B', fadeIn: 0.2})),
    ...SCORING_WHITE.map(([x,y], i) => ({at: 0.05 * (i + SCORING_BLACK.length), x, y, color: 'W', fadeIn: 0.2})),
  ];
  const annotations = [
    {at: 1.0, type: 'territory', owner: 'B', points: SCORING_B_TERR},
    {at: 1.2, type: 'territory', owner: 'W', points: SCORING_W_TERR},
    ...SCORING_DAME.map(p => ({at: 1.5, type: 'dame', x: p.x, y: p.y})),
  ];

  return (
    <>
      <SlideHeader num="05" title="Scoring & Komi" subtitle={
        <>Count stones on the board plus empty points not reaching opponent's color. <em>Komi</em> is a handicap added to White to offset Black's first-move advantage.</>
      } />
      <div style={{position:'absolute', left:BOARD_X, top:BOARD_Y}}>
        <GoAnim n={5} size={BOARD_SIZE} lt={lt} moves={moves} annotations={annotations} />
      </div>
      <ScoreTally lt={lt} />
    </>
  );
}

function ScoreTally({ lt }) {
  const panelOp = Easing.easeOutCubic(clamp((lt - 1.9) / 0.5, 0, 1));
  const bStonesOp = clamp((lt - 2.3) / 0.3, 0, 1);
  const bTerrOp   = clamp((lt - 2.8) / 0.3, 0, 1);
  const wStonesOp = clamp((lt - 3.5) / 0.3, 0, 1);
  const wTerrOp   = clamp((lt - 4.0) / 0.3, 0, 1);
  const komiOp    = clamp((lt - 5.2) / 0.3, 0, 1);
  const finalOp   = clamp((lt - 6.2) / 0.3, 0, 1);
  const winnerOp  = clamp((lt - 7.1) / 0.3, 0, 1);

  const bStones = Math.round(7 * clamp((lt - 2.3) / 0.6, 0, 1));
  const bTerr   = Math.round(3 * clamp((lt - 2.8) / 0.6, 0, 1));
  const wStones = Math.round(7 * clamp((lt - 3.5) / 0.6, 0, 1));
  const wTerr   = Math.round(3 * clamp((lt - 4.0) / 0.6, 0, 1));
  const komi    = clamp((lt - 5.2) / 0.5, 0, 1) * 7.5;
  const bFinal = 10;
  const wFinal = 17.5;

  const row = (label, val, opacity, mute = false) => (
    <div style={{
      display:'flex', justifyContent:'space-between',
      fontFamily:'var(--mono)', fontSize:14, padding:'4px 0',
      opacity, color: mute ? 'var(--ink-soft)' : 'var(--ink)',
      fontVariantNumeric:'tabular-nums',
    }}>
      <span>{label}</span>
      <span>{val}</span>
    </div>
  );

  return (
    <div style={{
      position:'absolute', left:620, top:210, width:500,
      fontFamily:'var(--serif)', color:'var(--ink)', opacity: panelOp,
    }}>
      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:32}}>
        <div>
          <div style={{display:'flex', alignItems:'center', gap:10, marginBottom:10}}>
            <div style={{width:14, height:14, borderRadius:'50%', background:'var(--stone-black)'}} />
            <div style={{fontFamily:'var(--mono)', fontSize:11, letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--ink-soft)'}}>Black</div>
          </div>
          {row('stones',    bStones,         bStonesOp)}
          {row('territory', `+${bTerr}`,    bTerrOp)}
          <div style={{
            borderTop:'1px solid rgba(31,26,20,0.18)', marginTop:8, paddingTop:6,
            display:'flex', justifyContent:'space-between',
            fontFamily:'var(--mono)', fontSize:18, opacity: finalOp, fontVariantNumeric:'tabular-nums',
          }}>
            <span style={{color:'var(--ink-soft)'}}>total</span>
            <span style={{fontWeight:600}}>{bFinal.toFixed(1)}</span>
          </div>
        </div>
        <div>
          <div style={{display:'flex', alignItems:'center', gap:10, marginBottom:10}}>
            <div style={{width:14, height:14, borderRadius:'50%', background:'var(--stone-white)', border:'1px solid rgba(60,40,20,0.5)'}} />
            <div style={{fontFamily:'var(--mono)', fontSize:11, letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--ink-soft)'}}>White</div>
          </div>
          {row('stones',    wStones,      wStonesOp)}
          {row('territory', `+${wTerr}`,  wTerrOp)}
          <div style={{
            display:'flex', justifyContent:'space-between',
            fontFamily:'var(--mono)', fontSize:14, padding:'4px 0',
            opacity: komiOp, color:'var(--accent-mcts)', fontVariantNumeric:'tabular-nums',
          }}>
            <span>komi</span>
            <span>+{komi.toFixed(1)}</span>
          </div>
          <div style={{
            borderTop:'1px solid rgba(31,26,20,0.18)', marginTop:8, paddingTop:6,
            display:'flex', justifyContent:'space-between',
            fontFamily:'var(--mono)', fontSize:18, opacity: finalOp, fontVariantNumeric:'tabular-nums',
          }}>
            <span style={{color:'var(--ink-soft)'}}>total</span>
            <span style={{fontWeight:600}}>{wFinal.toFixed(1)}</span>
          </div>
        </div>
      </div>

      <div style={{
        marginTop:24, padding:'14px 18px',
        background:'var(--accent-mcts-bg)', borderRadius:8,
        opacity: winnerOp,
        fontFamily:'var(--serif)', fontSize:16, lineHeight:1.45,
      }}>
        <div style={{fontFamily:'var(--mono)', fontSize:10, letterSpacing:'0.12em', textTransform:'uppercase', color:'var(--accent-mcts)', marginBottom:4}}>Result</div>
        White wins by <span style={{fontFamily:'var(--mono)', fontWeight:600}}>7.5</span>. Komi made the difference.
      </div>
    </div>
  );
}

// ── P1_Opening ──────────────────────────────────────────────────────────────
function P1_Opening() {
  const { localTime: lt } = useSprite();
  const moves = [
    { at: 0.4, x: 1, y: 1, color: 'B', number: 1 },
    { at: 1.3, x: 3, y: 3, color: 'W', number: 2 },
    { at: 2.2, x: 3, y: 1, color: 'B', number: 3 },
    { at: 3.1, x: 1, y: 3, color: 'W', number: 4 },
  ];
  const annotations = [
    {at: 4.0, until: 7.2, type:'tip', x:1, y:1, text:'corner · two walls free'},
    {at: 5.0, until: 7.2, type:'territory', owner:'B', points:[{x:0,y:0},{x:0,y:1},{x:1,y:0}]},
    {at: 5.3, until: 7.2, type:'territory', owner:'W', points:[{x:4,y:4},{x:3,y:4},{x:4,y:3}]},
  ];

  return (
    <>
      <SlideHeader num="07" title="Opening Moves" subtitle={
        <>Early stones work best near the corners. The board edges already form walls, so corner stones claim territory with the fewest friends.</>
      } />
      <div style={{position:'absolute', left:BOARD_X, top:BOARD_Y}}>
        <GoAnim n={5} size={BOARD_SIZE} lt={lt} moves={moves} annotations={annotations} />
      </div>
      <div style={{position:'absolute', left:620, top:240, width:500, fontFamily:'var(--serif)'}}>
        <Caption size={15} color="var(--ink)" style={{lineHeight:1.6, marginBottom:18}}>
          On a 5×5, the <em>3-3 points</em> at <span style={{fontFamily:'var(--mono)'}}>(1,1)</span> and
          <span style={{fontFamily:'var(--mono)'}}> (3,3)</span> are strong early claims.
        </Caption>
        <div style={{opacity: fadeIn(lt, 4.0, 0.5)}}>
          <div style={{fontFamily:'var(--mono)', fontSize:11, color:'var(--ink-soft)', letterSpacing:'0.1em', textTransform:'uppercase', marginBottom:8}}>Why corners?</div>
          <ul style={{fontFamily:'var(--serif)', fontSize:14, color:'var(--ink-soft)', lineHeight:1.65, paddingLeft:20, margin:0}}>
            <li>Two sides are already enclosed by the edge</li>
            <li>A handful of stones lock in real territory</li>
            <li>The centre is open — costly to defend, hard to hold</li>
          </ul>
        </div>
      </div>
    </>
  );
}

// ── P1_Capture ──────────────────────────────────────────────────────────────
function P1_Capture() {
  const { localTime: lt } = useSprite();
  const moves = [
    { at: 0.3, x: 2, y: 2, color: 'W' },
    { at: 1.5, x: 2, y: 1, color: 'B', number: 1 },
    { at: 2.7, x: 1, y: 2, color: 'B', number: 2 },
    { at: 3.9, x: 3, y: 2, color: 'B', number: 3 },
    { at: 5.1, x: 2, y: 3, color: 'B', number: 4 },
  ];
  const captures = [{ at: 5.5, x: 2, y: 2, fadeOut: 0.45 }];
  const annotations = [
    {at: 0.8, until: 1.5, type:'liberty', x:2, y:1},
    {at: 0.8, until: 2.7, type:'liberty', x:1, y:2},
    {at: 0.8, until: 3.9, type:'liberty', x:3, y:2},
    {at: 0.8, until: 5.1, type:'liberty', x:2, y:3},
    {at: 0.9, until: 2.4, type:'tip', x:2, y:2, text:'4 liberties', color:'var(--accent-net)'},
    {at: 2.9, until: 3.6, type:'tip', x:2, y:2, text:'3 liberties', color:'var(--accent-net)'},
    {at: 4.1, until: 4.8, type:'tip', x:2, y:2, text:'2 liberties', color:'var(--accent-net)'},
    {at: 4.3, until: 5.5, type:'atari', x:2, y:2},
    {at: 5.2, until: 5.5, type:'tip', x:2, y:2, text:'atari — 1 left'},
  ];

  return (
    <>
      <SlideHeader num="02" title="Capture" subtitle={
        <>A stone's <em>liberties</em> are the empty points directly next to it. Fill every last liberty of an enemy group and it comes off the board.</>
      } />
      <div style={{position:'absolute', left:BOARD_X, top:BOARD_Y}}>
        <GoAnim n={5} size={BOARD_SIZE} lt={lt} moves={moves} captures={captures} annotations={annotations} />
      </div>
      <div style={{position:'absolute', left:620, top:240, width:500, fontFamily:'var(--serif)', fontSize:15, color:'var(--ink)', lineHeight:1.65}}>
        <div style={{opacity: fadeIn(lt, 0.9)}}>
          A lone stone in the centre has <span style={{fontFamily:'var(--mono)', color:'var(--accent-net)'}}>4</span> liberties — one in each direction.
        </div>
        <div style={{marginTop:16, opacity: fadeIn(lt, 4.3), color:'var(--accent-mcts)'}}>
          Down to one → <em>atari</em>, the Go equivalent of a king in check.
        </div>
        <div style={{marginTop:16, opacity: fadeIn(lt, 5.6)}}>
          Fill the final liberty and the stone is removed from the board.
        </div>
        <div style={{marginTop:18, opacity: fadeIn(lt, 6.4), fontStyle:'italic', fontSize:13, color:'var(--ink-soft)', lineHeight:1.55}}>
          Suicide rule: you can't play a stone where it would have zero liberties — unless the move captures enemy stones first.
        </div>
      </div>
    </>
  );
}

// ── P1_Eyes ─────────────────────────────────────────────────────────────────
// Alive group on the top edge with two real eyes.
const EYES_POINTS = [
  {x:0,y:0},{x:2,y:0},{x:4,y:0},
  {x:0,y:1},{x:1,y:1},{x:2,y:1},{x:3,y:1},{x:4,y:1},
];
function P1_Eyes() {
  const { localTime: lt } = useSprite();
  const moves = EYES_POINTS.map((p, i) => ({
    at: 0.15 + i * 0.12, x: p.x, y: p.y, color: 'B',
  }));
  const annotations = [
    {at: 1.6, type:'group', points: EYES_POINTS, color:'var(--accent-net-soft)'},
    {at: 2.2, type:'eye', x:1, y:0},
    {at: 2.6, type:'eye', x:3, y:0},
    {at: 3.0, type:'tip', x:1, y:0, text:'real eye', color:'var(--accent-net)'},
    {at: 3.2, type:'tip', x:3, y:0, text:'real eye', color:'var(--accent-net)'},
    {at: 5.0, type:'ban', x:1, y:0},
    {at: 5.4, type:'ban', x:3, y:0},
  ];

  return (
    <>
      <SlideHeader num="06" title="Two Eyes" subtitle={
        <>A group with two independent internal empty points — two <em>eyes</em> — can never be captured. Filling either eye would be suicide, and the opponent can't fill both at once.</>
      } />
      <div style={{position:'absolute', left:BOARD_X, top:BOARD_Y}}>
        <GoAnim n={5} size={BOARD_SIZE} lt={lt} moves={moves} annotations={annotations} />
      </div>
      <div style={{position:'absolute', left:620, top:240, width:500}}>
        <Caption size={15} color="var(--ink)" style={{lineHeight:1.65, marginBottom:18}}>
          Corner and edge plays are cheap — the board itself serves as a wall. This eight-stone chain along the top encloses two independent eyes with the minimum of stones.
        </Caption>
        <div style={{opacity: fadeIn(lt, 4.8, 0.5)}}>
          <div style={{fontFamily:'var(--mono)', fontSize:11, color:'var(--accent-mcts)', letterSpacing:'0.1em', textTransform:'uppercase', marginBottom:6}}>Forbidden</div>
          <div style={{fontFamily:'var(--serif)', fontSize:14, color:'var(--ink-soft)', lineHeight:1.6}}>
            White can't legally enter either eye: a stone placed there has no liberties and captures nothing — that's suicide.
          </div>
        </div>
      </div>
    </>
  );
}

// ── P1_LifeDeath ────────────────────────────────────────────────────────────
// Two-board comparison: alive (2 eyes) vs. false-eye example (dead).
function P1_LifeDeath() {
  const { localTime: lt } = useSprite();

  // Left: alive (same shape as P1_Eyes but smaller)
  const aliveBlack = [
    {x:0,y:0},{x:2,y:0},{x:4,y:0},
    {x:0,y:1},{x:1,y:1},{x:2,y:1},{x:3,y:1},{x:4,y:1},
  ];
  const aliveMoves = aliveBlack.map((p, i) => ({at: 0.05 * i, x: p.x, y: p.y, color: 'B', fadeIn: 0.2}));
  const aliveAnnot = [
    {at: 0.6, type:'eye', x:1, y:0},
    {at: 0.8, type:'eye', x:3, y:0},
    {at: 1.4, type:'group', points: aliveBlack, color:'var(--accent-net-soft)'},
  ];

  // Right: false-eye shape. Black group has one real eye at (1,0) and one
  // "eye-shaped" but fragile point at (1,1) with an enemy diagonal at (2,2).
  const deadBlack = [
    {x:0,y:0},{x:2,y:0},
    {x:0,y:1},{x:1,y:1},{x:2,y:1},
    {x:1,y:2},
  ];
  const deadMoves = deadBlack.map((p, i) => ({at: 0.05 * i, x: p.x, y: p.y, color: 'B', fadeIn: 0.2}));
  const deadMoves2 = [
    ...deadMoves,
    {at: 2.2, x: 2, y: 2, color: 'W'},
  ];
  const deadAnnot = [
    {at: 0.6, type:'eye', x:1, y:0, color:'var(--accent-net)'},
    {at: 0.8, type:'tip', x:1, y:0, text:'real', color:'var(--accent-net)'},
    {at: 2.6, type:'eye', x:0, y:2, color:'var(--accent-mcts)'},
    {at: 2.8, type:'tip', x:0, y:2, text:'false eye', color:'var(--accent-mcts)'},
    {at: 3.4, type:'circle', x: 2, y: 2, radius: 0.75, strokeWidth: 1.6, dashed: true, color:'var(--accent-mcts)'},
  ];

  return (
    <>
      <SlideHeader num="05" title="Life & Death" subtitle={
        <>Not every eye-shaped hole is a real eye. A <em>false eye</em> can be broken by an enemy diagonal — a group with one real eye + one false eye can't form two and is declared <em>dead</em>.</>
      } />

      <div style={{position:'absolute', left:200, top:240}}>
        <GoAnim n={5} size={280} lt={lt} moves={aliveMoves} annotations={aliveAnnot} />
        <div style={{textAlign:'center', marginTop:12, fontFamily:'var(--mono)', fontSize:11, letterSpacing:'0.08em', textTransform:'uppercase', color:'var(--accent-net)'}}>Alive · 2 real eyes</div>
      </div>

      <div style={{position:'absolute', left:520, top:240}}>
        <GoAnim n={5} size={280} lt={lt} moves={deadMoves2} annotations={deadAnnot} />
        <div style={{textAlign:'center', marginTop:12, fontFamily:'var(--mono)', fontSize:11, letterSpacing:'0.08em', textTransform:'uppercase', color:'var(--accent-mcts)'}}>Dead · 1 real + 1 false</div>
      </div>

      <div style={{position:'absolute', left:840, top:260, width:340, fontFamily:'var(--serif)', color:'var(--ink)', fontSize:14, lineHeight:1.65}}>
        <div style={{opacity: fadeIn(lt, 1.5)}}>
          A <em>false eye</em> is an empty point that <strong>looks</strong> enclosed but whose diagonal is held by the enemy. Under attack, the enclosing stones collapse and the "eye" fills in.
        </div>
        <div style={{opacity: fadeIn(lt, 4.0), marginTop: 16, padding: '12px 14px', background:'rgba(31,26,20,0.05)', borderRadius: 6}}>
          A group with no way to make two eyes is already dead — at game end, its stones are removed and the area counts as the opponent's territory, even without playing out the capture.
        </div>
      </div>
    </>
  );
}

// ── P1_Liberties ────────────────────────────────────────────────────────────
function P1_Liberties() {
  const { localTime: lt } = useSprite();
  const moves = [
    { at: 0.2, x: 2, y: 2, color: 'B' },
    { at: 2.4, x: 2, y: 1, color: 'W' },
    { at: 3.6, x: 1, y: 2, color: 'W' },
    { at: 4.8, x: 3, y: 2, color: 'W' },
  ];
  const annotations = [
    {at: 0.6, until: 2.4, type:'liberty', x:2, y:1},
    {at: 0.6, until: 3.6, type:'liberty', x:1, y:2},
    {at: 0.6, until: 4.8, type:'liberty', x:3, y:2},
    {at: 0.6,               type:'liberty', x:2, y:3},
    {at: 0.8, until: 2.4, type:'tip', x:2, y:2, text:'4 liberties', color:'var(--accent-net)'},
    {at: 2.6, until: 3.6, type:'tip', x:2, y:2, text:'3 liberties', color:'var(--accent-net)'},
    {at: 3.8, until: 4.8, type:'tip', x:2, y:2, text:'2 liberties', color:'var(--accent-net)'},
    {at: 5.0, type:'atari', x:2, y:2},
    {at: 5.3, type:'tip', x:2, y:2, text:'atari — 1 left'},
    // Suicide demonstration (fade in late)
    {at: 6.3, type:'ban', x:2, y:3, color:'var(--accent-mcts)'},
    {at: 6.8, type:'tip', x:2, y:3, text:'suicide — forbidden'},
  ];

  return (
    <>
      <SlideHeader num="06" title="Liberties & Atari" subtitle={
        <>Every stone has <em>liberties</em> — the empty points touching it. When only one is left, the stone is in <em>atari</em>, one move from capture. You can't play into zero liberties unless your move captures first.</>
      } />
      <div style={{position:'absolute', left:BOARD_X, top:BOARD_Y}}>
        <GoAnim n={5} size={BOARD_SIZE} lt={lt} moves={moves} annotations={annotations} />
      </div>
      <div style={{position:'absolute', left:620, top:240, width:500, fontFamily:'var(--serif)'}}>
        <div style={{opacity: fadeIn(lt, 0.9), fontSize: 15, color: 'var(--ink)', lineHeight: 1.65, marginBottom: 16}}>
          A lone centre stone has <span style={{fontFamily:'var(--mono)', color:'var(--accent-net)'}}>4</span> liberties — one in each direction.
        </div>
        <div style={{opacity: fadeIn(lt, 5.0), padding:'12px 16px', background:'var(--accent-mcts-bg)', borderRadius: 8}}>
          <div style={{fontFamily:'var(--mono)', fontSize:11, color:'var(--accent-mcts)', letterSpacing:'0.1em', textTransform:'uppercase', marginBottom:4}}>Atari</div>
          <div style={{fontSize:14, color:'var(--ink)', lineHeight:1.6}}>
            One liberty left: the equivalent of a king in check. The next move on that liberty captures the stone.
          </div>
        </div>
        <div style={{opacity: fadeIn(lt, 6.6), marginTop: 14, fontStyle:'italic', fontSize:13, color:'var(--ink-soft)', lineHeight: 1.55}}>
          Suicide rule: you can't play a stone where it would have zero liberties — unless the move captures an enemy group first.
        </div>
      </div>
    </>
  );
}

// ── P1_Ko ───────────────────────────────────────────────────────────────────
function P1_Ko() {
  const { localTime: lt } = useSprite();
  const moves = [
    { at: 0.1, x: 2, y: 1, color: 'B' },
    { at: 0.3, x: 1, y: 2, color: 'B' },
    { at: 0.5, x: 2, y: 3, color: 'B' },
    { at: 0.7, x: 3, y: 1, color: 'W' },
    { at: 0.9, x: 2, y: 2, color: 'W' },
    { at: 1.1, x: 4, y: 2, color: 'W' },
    { at: 1.3, x: 3, y: 3, color: 'W' },
    // Black plays into a spot with 0 liberties — legal because it captures.
    { at: 2.8, x: 3, y: 2, color: 'B', number: 1 },
  ];
  const captures = [{ at: 3.2, x: 2, y: 2, fadeOut: 0.35 }];
  const annotations = [
    // (3,2) itself would be suicide — highlight before Black plays.
    {at: 1.8, until: 2.8, type:'circle', x:3, y:2, radius: 0.95, strokeWidth: 1.8, dashed: true, color:'var(--accent-mcts)'},
    {at: 1.8, until: 2.8, type:'tip',    x:3, y:2, text:'0 liberties — legal only if it captures', color:'var(--accent-mcts)'},
    // White (2,2) is in atari the whole setup.
    {at: 1.8, until: 2.8, type:'atari', x:2, y:2},
    // After capture, the empty (2,2) becomes the ko point that White cannot immediately retake.
    {at: 3.6, type:'ko',   x:2, y:2},
    {at: 4.0, until: 4.6, type:'tip',  x:2, y:2, text:'ko'},
    {at: 4.6, type:'ban',  x:2, y:2},
    {at: 4.9, type:'tip',  x:2, y:2, text:'White can\'t retake yet'},
  ];

  return (
    <>
      <SlideHeader num="03" title="The Ko Rule" />
      <div style={{position:'absolute', left:BOARD_X, top:BOARD_Y}}>
        <GoAnim n={5} size={BOARD_SIZE} lt={lt} moves={moves} captures={captures} annotations={annotations} />
      </div>
      <div style={{position:'absolute', left:620, top:240, width:500, fontFamily:'var(--serif)', color:'var(--ink)', lineHeight:1.65}}>
        <div style={{opacity: fadeIn(lt, 1.8), fontSize: 15, marginBottom: 14}}>
          The marked empty point has no liberties on its own — every neighbour is White. A stone there would normally be suicide.
        </div>
        <div style={{opacity: fadeIn(lt, 3.3), fontSize: 15, marginBottom: 14}}>
          But White <span style={{fontFamily:'var(--mono)'}}>(2,2)</span> is in atari, and Black's move captures it — so the move is legal, and Black's new stone <em>gains</em> a liberty from the capture.
        </div>
        <div style={{opacity: fadeIn(lt, 4.6), padding:'12px 16px', background:'var(--accent-mcts-bg)', borderRadius: 8, fontSize: 14}}>
          <div style={{fontFamily:'var(--mono)', fontSize:11, color:'var(--accent-mcts)', letterSpacing:'0.1em', textTransform:'uppercase', marginBottom:4}}>Ko ban</div>
          Afterwards, White can't immediately recapture at the marked point — that would restore the exact previous position. White must play elsewhere first, then consider retaking.
        </div>
      </div>
    </>
  );
}

// ── P1_Connectivity ─────────────────────────────────────────────────────────
function P1_Connectivity() {
  const { localTime: lt } = useSprite();

  const bambooMoves = [
    { at: 0.1, x: 1, y: 1, color: 'B' },
    { at: 0.25, x: 2, y: 1, color: 'B' },
    { at: 0.4, x: 1, y: 3, color: 'B' },
    { at: 0.55, x: 2, y: 3, color: 'B' },
    { at: 2.2, x: 1, y: 2, color: 'W' },
    { at: 3.1, x: 2, y: 2, color: 'B' },
  ];
  const bambooAnnot = [
    {at: 1.1, until: 2.1, type:'tip', x:1, y:2, text:'cut here?'},
    {at: 3.6, type:'group', points: [{x:1,y:1},{x:2,y:1},{x:2,y:2},{x:1,y:3},{x:2,y:3}], color:'var(--accent-net-soft)'},
  ];

  const keimaMoves = [
    { at: 0.1, x: 1, y: 1, color: 'B' },
    { at: 0.35, x: 2, y: 3, color: 'B' },
    { at: 2.2, x: 1, y: 2, color: 'W' },
  ];
  const keimaAnnot = [
    {at: 2.8, type:'group', points: [{x:1,y:1}], color:'rgba(47,120,180,0.45)'},
    {at: 2.8, type:'group', points: [{x:2,y:3}], color:'rgba(200,110,50,0.45)'},
    {at: 3.4, type:'tip', x:1, y:2, text:'cut!'},
  ];

  return (
    <>
      <SlideHeader num="08" title="Connect & Cut" subtitle={
        <>Stones are strong in groups. <em>Bamboo joints</em> are uncuttable; <em>knight's moves</em> trade solidity for speed — but a single move between them can slice them apart.</>
      } />
      <div style={{position:'absolute', left:200, top:250}}>
        <GoAnim n={5} size={260} lt={lt} moves={bambooMoves} annotations={bambooAnnot} />
        <div style={{marginTop:14, textAlign:'center', width:260}}>
          <div style={{fontFamily:'var(--mono)', fontSize:11, color:'var(--accent-net)', letterSpacing:'0.08em', textTransform:'uppercase'}}>Bamboo joint</div>
          <div style={{fontFamily:'var(--serif)', fontSize:13, color:'var(--ink-soft)', marginTop:6, lineHeight:1.55}}>
            Every cut attempt lets the defender connect on the other side to form atari.
          </div>
        </div>
      </div>

      <div style={{position:'absolute', left:500, top:250}}>
        <GoAnim n={5} size={260} lt={lt} moves={keimaMoves} annotations={keimaAnnot} />
        <div style={{marginTop:14, textAlign:'center', width:260}}>
          <div style={{fontFamily:'var(--mono)', fontSize:11, color:'var(--accent-mcts)', letterSpacing:'0.08em', textTransform:'uppercase'}}>Knight's move (keima)</div>
          <div style={{fontFamily:'var(--serif)', fontSize:13, color:'var(--ink-soft)', marginTop:6, lineHeight:1.55}}>
            Efficient territory expansion — but easier for opponent to "cut" the pair
          </div>
        </div>
      </div>

      <div style={{position:'absolute', left:800, top:270, width:360, fontFamily:'var(--serif)', fontSize:14, color:'var(--ink)', lineHeight:1.65}}>
        Strong players think in <em>groups</em>, not individual stones. Whether a shape is cuttable often decides the game — once a group is severed, the pieces each need their own two eyes to live.
      </div>
    </>
  );
}

// ── P1_EndOfGame ────────────────────────────────────────────────────────────
function P1_EndOfGame() {
  const { localTime: lt } = useSprite();
  const moves = [
    ...SCORING_BLACK.map(([x,y], i) => ({at: 0.05 * i,                x, y, color: 'B', fadeIn: 0.18})),
    ...SCORING_WHITE.map(([x,y], i) => ({at: 0.05 * (i + SCORING_BLACK.length), x, y, color: 'W', fadeIn: 0.18})),
  ];
  const annotations = [
    {at: 3.6, type:'territory', owner:'B', points: SCORING_B_TERR},
    {at: 3.7, type:'territory', owner:'W', points: SCORING_W_TERR},
    ...SCORING_DAME.map(p => ({at: 4.0, type: 'dame', x: p.x, y: p.y})),
  ];

  const passOp = fadeIn(lt, 1.4, 0.45);
  const pass2Op = fadeIn(lt, 2.2, 0.45);
  const endOp  = fadeIn(lt, 3.2, 0.45);
  const scoreOp = fadeIn(lt, 4.6, 0.5);
  const ruleOp  = fadeIn(lt, 5.5, 0.5);

  return (
    <>
      <SlideHeader num="04" title="Ending the Game" subtitle={
        <>When neither player can gain territory or captures, they <em>pass</em>. Two consecutive passes end the game. Then the board is counted.</>
      } />
      <div style={{position:'absolute', left:BOARD_X, top:BOARD_Y}}>
        <GoAnim n={5} size={BOARD_SIZE} lt={lt} moves={moves} annotations={annotations} />
      </div>

      {/* Pass card animations */}
      <div style={{
        position:'absolute', left:620, top:230, width:500,
      }}>
        <div style={{
          opacity: passOp, transform: `translateX(${(1 - passOp) * 12}px)`,
          display:'flex', alignItems:'center', gap:14, marginBottom:10,
          fontFamily:'var(--serif)', fontSize:17, color:'var(--ink)',
        }}>
          <div style={{width:18, height:18, borderRadius:'50%', background:'var(--stone-black)'}} />
          <span>Black: <span style={{fontFamily:'var(--mono)', color:'var(--ink-soft)'}}>pass</span></span>
        </div>
        <div style={{
          opacity: pass2Op, transform: `translateX(${(1 - pass2Op) * 12}px)`,
          display:'flex', alignItems:'center', gap:14, marginBottom:18,
          fontFamily:'var(--serif)', fontSize:17, color:'var(--ink)',
        }}>
          <div style={{width:18, height:18, borderRadius:'50%', background:'var(--stone-white)', border:'1px solid rgba(60,40,20,0.5)'}} />
          <span>White: <span style={{fontFamily:'var(--mono)', color:'var(--ink-soft)'}}>pass</span></span>
        </div>

        <div style={{opacity: endOp, fontFamily:'var(--mono)', fontSize:11, letterSpacing:'0.14em', textTransform:'uppercase', color:'var(--accent-mcts)', marginBottom:14}}>
          Game ends — count the board
        </div>

        <div style={{
          opacity: scoreOp, padding:'14px 18px',
          background:'rgba(31,26,20,0.05)', borderRadius: 8,
          fontFamily:'var(--serif)', fontSize:15, color:'var(--ink)', lineHeight:1.55,
        }}>
          Stones + enclosed empty points + komi. The higher total wins.
        </div>

        <div style={{
          opacity: ruleOp, marginTop: 12,
          fontFamily:'var(--serif)', fontSize:13, color:'var(--ink-soft)',
          lineHeight:1.55, fontStyle:'italic',
        }}>
          We use <em><a href="https://www.cs.cmu.edu/~wjh/go/tmp/rules/TrompTaylor.html"
                          target="_blank" rel="noopener noreferrer">Tromp-Taylor</a></em> scoring (area = stones + territory) — simple and unambiguous scoring rules to implement for a Go AI. Modern form of Chinese area scoring.
        </div>
      </div>
    </>
  );
}

Object.assign(window, {
  P1_Title, P1_StrategiesTitle, P1_PlacingStones, P1_Scoring, P1_Opening,
  P1_Capture, P1_Eyes, P1_LifeDeath, P1_Liberties, P1_Ko, P1_Connectivity,
  P1_EndOfGame,
});
