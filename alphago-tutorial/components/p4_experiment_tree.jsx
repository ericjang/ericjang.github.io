// components/p4_experiment_tree.jsx
// Part 4 · slide 12 — experiment lineage tree.
// Horizontally scrollable diagram. Rows are research themes; nodes are
// individual investigations colored by outcome. Solid edges are primary
// child paths; dotted edges are related-parent / cross-branch links.

const TREE = {
  rows: [
    { id: 'foundations',            label: 'Architecture & data foundations' },
    { id: 'first_mcts',             label: 'First MCTS implementation' },
    { id: 'evaluation_diagnostics', label: 'Evaluation diagnostics' },
    { id: 'recovery',               label: 'Recovery training fork' },
    { id: 'iterative_self_play',    label: 'Iterative self-play' },
    { id: 'throughput_infra',       label: 'Throughput / collection infra' },
    { id: 'sync_rl',                label: 'MCTS iter. with synchronous RL' },
    { id: 'offline_mcts',           label: 'Batch MCTS on offline data' },
    { id: 'nineteen',               label: 'Scaling up to 19×19' },
  ],
  dates: [
    { x:   90, label: 'Dec 24' }, { x:  650, label: 'Dec 29' },
    { x: 1210, label: 'Jan 2'  }, { x: 1510, label: 'Jan 14' },
    { x: 1840, label: 'Jan 20' }, { x: 2380, label: 'Feb'    },
    { x: 2460, label: 'Feb 13' }, { x: 2700, label: 'Mar 29' },
    { x: 2940, label: 'Apr 3'  }, { x: 3420, label: 'Apr 6'  },
    { x: 3660, label: 'Apr 8'  }, { x: 3900, label: 'Apr 10' },
    { x: 4140, label: 'Apr 12' }, { x: 4620, label: 'Apr 18' },
    { x: 5100, label: 'Apr 20' }, { x: 5340, label: 'Apr 22' },
    { x: 5580, label: 'Apr 25' },
  ],
  nodes: [
    { id:'scale',            row:'foundations',            x:  90, date:'Dec 24-26',  title:'Scaling laws + wider models',           note:'Batch, Muon/AdamW, ResNet size, data scaling.',                              status:'mixed' },
    { id:'dedup',            row:'foundations',            x: 360, date:'Dec 26-29',  title:'Dataset dedup + variants',              note:'Normalized/five-way dedup clarified data quality.',                          status:'good'  },
    { id:'mup',              row:'foundations',            x: 640, date:'Dec 27-28',  title:'MuP and architecture search',           note:'MuP sweeps, Transformer vs ResNet, 100M baseline.',                          status:'mixed' },
    { id:'baseline',         row:'foundations',            x: 920, date:'Jan 2-14',   title:'KataGo distillation baseline',          note:'Offline KataGo data gave the durable teacher foundation.',                   status:'good'  },
    { id:'val_feb13',        row:'foundations',            x:2460, date:'Feb 13',     title:'Optimize val loss',                     note:'Early val-loss optimization branch after dataset scaling.',                  status:'mixed' },
    { id:'val_mar29',        row:'foundations',            x:2700, date:'Mar 29',     title:'Optimize val loss',                     note:'Restarted focused 9x9 val-loss search.',                                     status:'mixed' },
    { id:'val_apr03',        row:'foundations',            x:2940, date:'Apr 3',      title:'KataGo val-loss v9',                    note:'Teacher-data fitting became the main warm-start path.',                      status:'good'  },
    { id:'val_apr04',        row:'foundations',            x:3180, date:'Apr 4',      title:'Val-loss v10',                          note:'Refined the 9x9 checkpoint recipe.',                                         status:'good'  },
    { id:'val_apr06',        row:'foundations',            x:3420, date:'Apr 6',      title:'Val-loss v11 + transformer',            note:'ResNet path kept improving; transformer branch checked in parallel.',        status:'mixed' },
    { id:'val_apr07',        row:'foundations',            x:3660, date:'Apr 7',      title:'Val-loss v12',                          note:'Convnet/transformer comparison and tuned v12 checkpoint.',                   status:'good'  },
    { id:'val_apr10',        row:'foundations',            x:3900, date:'Apr 10',     title:'Final warm-start checkpoint',           note:'Best checkpoint used by LearnGo v2.',                                        status:'good'  },

    { id:'mcts_impl',        row:'first_mcts',             x: 650, date:'Dec 29',     title:'Policy-network MCTS',                   note:'Implemented and debugged MCTS over the policy/value net.',                   status:'good'  },
    { id:'mcts_eval',        row:'first_mcts',             x: 920, date:'Dec 30-31',  title:'MCTS vs raw policy',                    note:'MCTS beat raw policy; C++ parity and shared lib validated.',                 status:'good'  },
    { id:'mcts_data',        row:'first_mcts',             x:1210, date:'Jan 2-5',    title:'MCTS self-play data',                   note:'Collected MCTS, KataGo, combined game data; analyzed score/komi.',           status:'mixed' },
    { id:'mcts_tune',        row:'first_mcts',             x:1510, date:'Jan 14-16',  title:'MCTS tuning sweep',                     note:'Sims, cpuct, temperature, Dirichlet, color; final config emerged.',          status:'good'  },
    { id:'blunders',         row:'first_mcts',             x:1790, date:'Jan 16-17',  title:'Blunder and pass analysis',             note:'Found late-game/pass/action issues and training cutoffs.',                   status:'mixed' },

    { id:'scoring_diag',     row:'evaluation_diagnostics', x:1040, date:'Jan 5-8',    title:'Scoring & max-move diagnostics',        note:'Random-white upsets traced to premature max-move cutoffs and komi.',         status:'mixed' },
    { id:'color_diag',       row:'evaluation_diagnostics', x:1510, date:'Jan 14',     title:'Color asymmetry diagnostics',           note:'Black/white WR gap and KataGo matchup asymmetry became recurring signals.',  status:'mixed' },
    { id:'pass_diag',        row:'evaluation_diagnostics', x:1790, date:'Jan 16-17',  title:'Pass/endgame diagnostics',              note:'Blunder, pass-action, and late-game analyses exposed endgame weaknesses.',   status:'mixed' },

    { id:'adv',              row:'recovery',               x: 650, date:'Dec 29',     title:'Advantage / skill conditioning',        note:'Conditioning variants explored, but signal remained fragile.',               status:'bad'   },
    { id:'recovery0',        row:'recovery',               x:1540, date:'Jan 17-19',  title:'Re-do MCTS recovery data',              note:'Recovery examples looked useful; from-scratch loop collapsed.',              status:'bad'   },
    { id:'recovery_fine',    row:'recovery',               x:1840, date:'Jan 20-22',  title:'Fine-tuned recovery',                   note:'5 alternates helped; 10 alternates diluted quality and regressed.',          status:'mixed' },

    { id:'iter0',            row:'iterative_self_play',    x: 870, date:'Dec 29-Jan 3', title:'Early online self-play',              note:'Replay buffers and online training were valuable but unstable.',             status:'mixed' },
    { id:'datasetv3',        row:'iterative_self_play',    x:2120, date:'Jan 25-27',  title:'Dataset-v3 scaling',                    note:'10M/100M and checkpoint-pair eval clarified scale limits.',                  status:'mixed' },
    { id:'datasetv4',        row:'iterative_self_play',    x:2380, date:'Feb 1-5',    title:'Dataset-v4/v5 scaling',                 note:'Scaling and checkpoint-pair datasets improved the training substrate.',      status:'good'  },
    { id:'selfplay7m',       row:'iterative_self_play',    x:2660, date:'Feb 6-8',    title:'7M iterative self-play',                note:'All-checkpoint eval fix produced real WR gains, but noisy strength.',        status:'good'  },
    { id:'strategy',         row:'iterative_self_play',    x:2940, date:'Feb 8-10',   title:'Strategy landscape + continuous mixes', note:'Landscape analysis useful; weighted/uniform continuous mixes degraded.',     status:'mixed' },

    { id:'collector',        row:'throughput_infra',       x:2160, date:'Jan 20-21',  title:'Collector scaling',                     note:'Distributed/local inference servers made collection more practical.',        status:'good'  },
    { id:'throughput_opt',   row:'throughput_infra',       x:3420, date:'Apr 11-13',  title:'Batched inference and PCR sweeps',      note:'Batched inference gave ~4x throughput; PCR 80/18/2 was the collect winner.', status:'good'  },
    { id:'label_throughput', row:'throughput_infra',       x:4620, date:'Apr 19',     title:'MCTS label throughput',                 note:'Measured board-labeling parallelism before offline self-distillation.',      status:'good'  },
    { id:'throughput',       row:'throughput_infra',       x:5580, date:'Apr 25',     title:'Throughput and web cluster bench',      note:'Benchmarked MCTS throughput and web-demo serving path.',                     status:'good'  },

    { id:'learn0',           row:'sync_rl',                x:3660, date:'Apr 8-9',    title:'LearnGo v0/v1',                         note:'First loop variants exposed warm-start and LR sensitivity.',                 status:'mixed' },
    { id:'learn2',           row:'sync_rl',                x:3900, date:'Apr 10',     title:'LearnGo v2',                            note:'256ch warm start reached 34.3% WR, then plateaued.',                         status:'good'  },
    { id:'learn3',           row:'sync_rl',                x:4140, date:'Apr 12',     title:'LearnGo v3 dense loss',                 note:'Dense MCTS visits + low WD crossed 50%, peaked near 64% WR.',                status:'good'  },
    { id:'decent',           row:'sync_rl',                x:4380, date:'Apr 13-17',  title:'Local teacher → SSH v7',                note:'Decentralized collection made the strong 9x9 loop operational.',             status:'good'  },
    { id:'fast',             row:'sync_rl',                x:4620, date:'Apr 18',     title:'LearnGo fast',                          note:'Fewer games/iter and PCR tweaks improved per-game efficiency but stayed noisy.', status:'mixed' },

    { id:'selfdistill',      row:'offline_mcts',           x:4860, date:'Apr 19',     title:'MCTS self-distill',                     note:'Async offline labeling pipeline decoupled board collection from MCTS relabeling.', status:'mixed' },
    { id:'rootq',            row:'offline_mcts',           x:5100, date:'Apr 19-21',  title:'MCTS label + RootQ stabilization',      note:'2048 relabeling and RootQ became the offline recipe.',                       status:'good'  },
    { id:'offline',          row:'offline_mcts',           x:5340, date:'Apr 20-23',  title:'Offline MCTS labels vs on-policy',      note:'Offline labels improved white but made black volatile.',                     status:'mixed' },
    { id:'balanced',         row:'offline_mcts',           x:5580, date:'Apr 23',     title:'Balanced-value fork',                   note:'Isolates value-label balance from deep off-policy MCTS relabels.',           status:'mixed' },

    { id:'warm19',           row:'nineteen',               x:5340, date:'Apr 22',     title:'19×19 + 9×9 warm mix',                  note:'Warm-started size-invariant model with mixed board sizes.',                  status:'good'  },
    { id:'web19',            row:'nineteen',               x:5580, date:'Apr 24-25',  title:'Web-demo 19×19 model',                  note:'19×19+9×9 co-trained model crushed 9×9-only in head-to-head.',               status:'good'  },
  ],
  primaryEdges: [
    ['scale','dedup'], ['dedup','mup'], ['mup','baseline'], ['baseline','val_feb13'],
    ['val_feb13','val_mar29'], ['val_mar29','val_apr03'], ['val_apr03','val_apr04'],
    ['val_apr04','val_apr06'], ['val_apr06','val_apr07'], ['val_apr07','val_apr10'],
    ['mcts_impl','mcts_eval'], ['mcts_eval','mcts_data'], ['mcts_data','mcts_tune'],
    ['mcts_tune','blunders'], ['blunders','datasetv3'],
    ['mcts_data','scoring_diag'], ['scoring_diag','color_diag'], ['color_diag','pass_diag'],
    ['baseline','mcts_impl'], ['mcts_tune','recovery0'], ['recovery0','recovery_fine'],
    ['iter0','datasetv3'], ['datasetv3','datasetv4'], ['datasetv4','selfplay7m'], ['selfplay7m','strategy'],
    ['recovery_fine','collector'], ['collector','throughput_opt'], ['throughput_opt','label_throughput'],
    ['val_apr07','learn0'], ['val_apr10','learn2'], ['learn0','learn2'], ['learn2','learn3'],
    ['learn3','decent'], ['decent','fast'], ['fast','selfdistill'],
    ['label_throughput','selfdistill'], ['selfdistill','rootq'], ['rootq','offline'], ['offline','balanced'],
    ['decent','throughput'], ['warm19','web19'],
  ],
  relatedEdges: [
    ['pass_diag','blunders'], ['mcts_tune','learn3'], ['selfplay7m','learn3'],
    ['collector','decent'], ['throughput_opt','decent'], ['throughput_opt','fast'],
    ['label_throughput','rootq'], ['learn3','warm19'], ['val_apr10','warm19'],
    ['rootq','warm19'], ['warm19','throughput'],
  ],
};

const TREE_STATUS_COLOR = {
  good:  '#1f8a5b',
  mixed: '#b7791f',
  bad:   '#b64242',
};
const TREE_STATUS_LABEL = { good: 'Worked', mixed: 'Mixed', bad: 'Failed' };

// Layout constants for the diagram inside the scroll viewport.
// X_SCALE controls horizontal node spacing — at 0.6 the 168px-wide nodes
// were ~144px apart and overlapped; 1.0 gives ~72px gap between the
// densest adjacent pair (the val_apr* sequence at 240 raw-units apart).
const TREE_X_SCALE = 1.0;
const TREE_X_PAD   = 24;
const TREE_TOP_PAD = 36;
const TREE_DATE_Y  = 14;
const TREE_NODE_W  = 168;
const TREE_NODE_H  = 44;

// Pre-compute the time-axis (x) layout and adjacency once. The y layout is
// recomputed per-render from the measured viewport height so all rows fit.
const TREE_X = (() => {
  const xs = TREE.nodes.map(n => n.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const canvasW = Math.round((maxX - minX) * TREE_X_SCALE) + TREE_NODE_W + TREE_X_PAD * 2;

  const nodeX = Object.fromEntries(TREE.nodes.map(n => [
    n.id, TREE_X_PAD + (n.x - minX) * TREE_X_SCALE,
  ]));
  const dateXs = TREE.dates.map(d => ({
    label: d.label,
    px: TREE_X_PAD + (d.x - minX) * TREE_X_SCALE,
  }));

  const adj = {};
  for (const n of TREE.nodes) adj[n.id] = new Set();
  for (const [a, b] of TREE.primaryEdges) { adj[a]?.add(b); adj[b]?.add(a); }
  for (const [a, b] of TREE.relatedEdges) { adj[a]?.add(b); adj[b]?.add(a); }

  return { canvasW, nodeX, dateXs, adj };
})();

function P4_ExperimentTree() {
  const { localTime: lt } = useSprite();
  const headerOp = Easing.easeOutCubic(clamp(lt / 0.5, 0, 1));
  const bodyOp   = Easing.easeOutCubic(clamp((lt - 0.4) / 0.6, 0, 1));

  const [hovered, setHovered] = React.useState(null);
  const [dragging, setDragging] = React.useState(false);
  const scrollRef = React.useRef(null);
  const dragRef = React.useRef({ active: false, startX: 0, startScrollLeft: 0, moved: false });
  const userInteractedRef = React.useRef(false);

  // Auto-pan during the first ~5 s of the slide so viewers see the diagram
  // is wide. Stops once the user starts hovering or dragging so we don't
  // fight their input.
  React.useEffect(() => {
    const el = scrollRef.current;
    if (!el || hovered != null || userInteractedRef.current) return;
    if (lt > 5.5) return;
    const max = el.scrollWidth - el.clientWidth;
    if (max <= 0) return;
    const t = clamp((lt - 0.6) / 4.5, 0, 1);
    el.scrollLeft = Easing.easeInOutCubic(t) * max;
  }, [lt, hovered]);

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    if (e.target.closest('[data-tree-node]')) return;
    const el = scrollRef.current;
    if (!el) return;
    dragRef.current = {
      active: true, moved: false,
      startX: e.clientX, startScrollLeft: el.scrollLeft,
    };
    setDragging(true);
    userInteractedRef.current = true;
    el.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e) => {
    const d = dragRef.current;
    if (!d.active) return;
    const el = scrollRef.current;
    if (!el) return;
    const dx = e.clientX - d.startX;
    if (Math.abs(dx) > 2) d.moved = true;
    el.scrollLeft = d.startScrollLeft - dx;
  };
  const endDrag = (e) => {
    const d = dragRef.current;
    if (!d.active) return;
    d.active = false;
    setDragging(false);
    const el = scrollRef.current;
    el?.releasePointerCapture?.(e.pointerId);
  };

  // Measure viewport height so we can fit all rows without clipping.
  const [viewportH, setViewportH] = React.useState(0);
  React.useLayoutEffect(() => {
    if (!scrollRef.current) return;
    const el = scrollRef.current;
    const update = () => setViewportH(el.clientHeight);
    update();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const { canvasW, nodeX, dateXs, adj } = TREE_X;

  // Distribute rows evenly across the available viewport height. Reserve
  // space at the top for the date strip (TREE_TOP_PAD) and a small bottom
  // pad. If the viewport hasn't been measured yet (first paint), fall back
  // to a sensible default that still keeps all rows visible at typical sizes.
  const ROW_COUNT = TREE.rows.length;
  const BOTTOM_PAD = 12;
  const availH = Math.max(0, (viewportH || 480) - TREE_TOP_PAD - BOTTOM_PAD);
  const rowH = Math.max(36, availH / ROW_COUNT);
  const rowY = {};
  TREE.rows.forEach((r, i) => { rowY[r.id] = TREE_TOP_PAD + i * rowH; });
  const canvasH = TREE_TOP_PAD + ROW_COUNT * rowH + BOTTOM_PAD;

  const placed = TREE.nodes.map(n => ({
    ...n,
    px: nodeX[n.id],
    py: rowY[n.row],
  }));
  const byId = Object.fromEntries(placed.map(n => [n.id, n]));

  const hovNode = hovered ? byId[hovered] : null;

  const isEdgeActive = (a, b) =>
    hovered != null && (a === hovered || b === hovered);

  // Tooltip is positioned relative to the viewport, near the hovered node.
  let tipLeft = 0, tipTop = 0, tipFlipX = false;
  if (hovNode && scrollRef.current) {
    const rect = scrollRef.current.getBoundingClientRect();
    const sl = scrollRef.current.scrollLeft;
    const screenX = hovNode.px + TREE_NODE_W / 2 - sl;
    tipFlipX = screenX > rect.width / 2;
    tipLeft = tipFlipX
      ? hovNode.px - 280 - 8
      : hovNode.px + TREE_NODE_W + 8;
    tipTop  = hovNode.py - 6;
  }

  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', flexDirection: 'column',
      paddingLeft: 200, paddingRight: 60,
      paddingTop: 76, paddingBottom: 40,
      gap: 18,
    }}>
      {/* Header strip: title + subtitle on the left, legend on the right.
          Sized by content; the viewport below takes whatever is left over. */}
      <div style={{
        display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
        gap: 24, opacity: headerOp,
      }}>
        <div style={{ flex: '1 1 auto', minWidth: 0 }}>
          <SectionLabel num="17" title="Experiment lineage tree" />
          <div style={{
            marginTop: 12,
            fontFamily: 'var(--serif)', fontSize: 15,
            color: 'var(--ink-soft)',
            maxWidth: 920, lineHeight: 1.55,
          }}>
            R&amp;D can be a nonlinear process where we encounter several dead ends before getting things working. Each row is a research theme; each card is an experiments colored by outcome. Solid lines trace the primary experiment path; dotted lines show influence on later experiments.
          </div>
        </div>
        <div style={{
          flex: '0 0 auto',
          display: 'flex', gap: 14,
          fontFamily: 'var(--mono)', fontSize: 10.5,
          letterSpacing: '0.08em', textTransform: 'uppercase',
          color: 'var(--ink-soft)',
          paddingTop: 4,
        }}>
          {['good','mixed','bad'].map(s => (
            <span key={s} style={{ display:'inline-flex', alignItems:'center', gap:6 }}>
              <span style={{
                width: 10, height: 10, borderRadius: 5,
                border: `2px solid ${TREE_STATUS_COLOR[s]}`, background: 'var(--bg)',
              }} />
              {TREE_STATUS_LABEL[s]}
            </span>
          ))}
        </div>
      </div>

      {/* Scrollable diagram viewport — flex:1 fills remaining vertical space. */}
      <div
        ref={scrollRef}
        onWheel={(e) => {
          // Convert vertical wheel into horizontal pan so trackpad/scroll wheel
          // users can navigate the timeline without rebinding axes.
          const el = scrollRef.current;
          if (!el) return;
          if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
            el.scrollLeft += e.deltaY;
            e.preventDefault();
          }
          userInteractedRef.current = true;
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        style={{
          flex: '1 1 auto',
          minHeight: 0,
          position: 'relative',
          overflowX: 'auto', overflowY: 'hidden',
          background: 'rgba(31,26,20,0.025)',
          border: '1px solid rgba(31,26,20,0.10)',
          borderRadius: 6,
          opacity: bodyOp,
          cursor: dragging ? 'grabbing' : (hovered ? 'default' : 'grab'),
          userSelect: 'none',
          touchAction: 'pan-y',
        }}
      >
        <div style={{
          position: 'relative',
          width: canvasW, height: canvasH,
        }}>
          {/* Row bands */}
          {TREE.rows.map((r, i) => (
            <div key={r.id} style={{
              position:'absolute', left:0, top: rowY[r.id] - 6,
              width: canvasW, height: rowH,
              background: i % 2 === 0 ? 'rgba(31,26,20,0.018)' : 'transparent',
              borderTop: '1px solid rgba(31,26,20,0.06)',
            }} />
          ))}

          {/* Sticky row labels (left rail) */}
          {TREE.rows.map(r => (
            <div key={`lbl-${r.id}`} style={{
              position: 'sticky', left: 6, top: rowY[r.id] + 8,
              width: 188, height: 0, zIndex: 4,
              pointerEvents: 'none',
            }}>
              <div style={{
                display:'inline-block',
                padding:'4px 8px',
                background:'linear-gradient(90deg, var(--bg) 80%, rgba(247,243,234,0))',
                fontFamily:'var(--mono)', fontSize:10,
                letterSpacing:'0.06em', textTransform:'uppercase',
                color:'rgba(31,26,20,0.55)',
                fontWeight: 600,
                whiteSpace: 'nowrap',
              }}>
                {r.label}
              </div>
            </div>
          ))}

          {/* Date gridlines + labels */}
          {dateXs.map((d, i) => (
            <React.Fragment key={`dg-${i}`}>
              <div style={{
                position:'absolute', left: d.px, top: TREE_DATE_Y + 14, width: 1,
                height: canvasH - (TREE_DATE_Y + 14) - 4, background: 'rgba(31,26,20,0.08)',
              }} />
              <div style={{
                position:'absolute', left: d.px + 4, top: TREE_DATE_Y,
                fontFamily:'var(--mono)', fontSize: 9.5,
                letterSpacing:'0.06em', color:'rgba(31,26,20,0.45)',
                fontWeight: 600, whiteSpace:'nowrap',
              }}>
                {d.label}
              </div>
            </React.Fragment>
          ))}

          {/* Edges */}
          <svg style={{
            position:'absolute', inset: 0,
            width: canvasW, height: canvasH,
            pointerEvents: 'none', overflow: 'visible',
          }}>
            {TREE.primaryEdges.map(([a, b], i) => {
              const A = byId[a], B = byId[b];
              if (!A || !B) return null;
              const x1 = A.px + TREE_NODE_W,   y1 = A.py + TREE_NODE_H / 2;
              const x2 = B.px,                 y2 = B.py + TREE_NODE_H / 2;
              const dx = Math.max(40, Math.abs(x2 - x1) * 0.45);
              const active = isEdgeActive(a, b);
              const dim = hovered != null && !active;
              return (
                <path key={`pe-${i}`}
                  d={`M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`}
                  fill="none"
                  stroke={active ? 'var(--accent-mcts)' : 'rgba(52,66,86,0.55)'}
                  strokeWidth={active ? 2.4 : 1.6}
                  strokeLinecap="round"
                  opacity={dim ? 0.18 : 1}
                />
              );
            })}
            {TREE.relatedEdges.map(([a, b], i) => {
              const A = byId[a], B = byId[b];
              if (!A || !B) return null;
              const x1 = A.px + TREE_NODE_W,   y1 = A.py + TREE_NODE_H / 2;
              const x2 = B.px,                 y2 = B.py + TREE_NODE_H / 2;
              const dx = Math.max(40, Math.abs(x2 - x1) * 0.45);
              const active = isEdgeActive(a, b);
              const dim = hovered != null && !active;
              return (
                <path key={`re-${i}`}
                  d={`M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`}
                  fill="none"
                  stroke={active ? 'var(--accent-mcts)' : 'rgba(103,80,164,0.55)'}
                  strokeWidth={active ? 2.0 : 1.4}
                  strokeDasharray="5 6"
                  strokeLinecap="round"
                  opacity={dim ? 0.18 : 1}
                />
              );
            })}
          </svg>

          {/* Nodes */}
          {placed.map(n => {
            const isHov = hovered === n.id;
            const isAdj = hovered != null && adj[hovered]?.has(n.id);
            const dim = hovered != null && !isHov && !isAdj;
            const color = TREE_STATUS_COLOR[n.status];
            return (
              <button key={n.id}
                type="button"
                data-tree-node
                aria-label={`${n.date}: ${n.title}. ${TREE_STATUS_LABEL[n.status]}. ${n.note}`}
                onMouseEnter={() => { if (!dragRef.current.active) setHovered(n.id); }}
                onMouseLeave={() => setHovered(prev => (prev === n.id ? null : prev))}
                onFocus={() => setHovered(n.id)}
                onBlur={() => setHovered(prev => (prev === n.id ? null : prev))}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') {
                    e.currentTarget.blur();
                    setHovered(null);
                  }
                }}
                style={{
                  position:'absolute', left: n.px, top: n.py,
                  width: TREE_NODE_W, height: TREE_NODE_H,
                  padding: '6px 9px',
                  borderRadius: 5,
                  appearance: 'none',
                  WebkitAppearance: 'none',
                  textAlign: 'left',
                  background: 'var(--bg)',
                  border: '1px solid rgba(31,26,20,0.18)',
                  borderLeft: `5px solid ${color}`,
                  boxShadow: isHov
                    ? '0 6px 18px rgba(31,26,20,0.20)'
                    : '0 1px 2px rgba(31,26,20,0.06)',
                  opacity: dim ? 0.28 : 1,
                  transform: isHov ? 'translateY(-1px)' : 'none',
                  transition: 'opacity 140ms, box-shadow 140ms, transform 140ms',
                  cursor: 'pointer',
                  zIndex: isHov ? 6 : 2,
                  pointerEvents: 'auto',
                  font: 'inherit',
                }}>
                <div style={{
                  display:'flex', justifyContent:'space-between', alignItems:'center', gap: 6,
                  fontFamily:'var(--mono)', fontSize: 8.5,
                  letterSpacing:'0.06em', textTransform:'uppercase',
                  color:'var(--ink-soft)', marginBottom: 2,
                  whiteSpace:'nowrap', overflow:'hidden',
                }}>
                  <span style={{ overflow:'hidden', textOverflow:'ellipsis' }}>{n.date}</span>
                  <span style={{
                    background: color, color: 'var(--bg)',
                    padding:'1px 5px', borderRadius: 999,
                    fontSize: 8, letterSpacing: 0, textTransform:'none',
                    fontWeight: 600,
                  }}>
                    {TREE_STATUS_LABEL[n.status].toLowerCase()}
                  </span>
                </div>
                <div style={{
                  fontFamily:'var(--serif)', fontSize: 11.5,
                  fontWeight: 600, color: 'var(--ink)',
                  lineHeight: 1.18,
                  whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis',
                }}>
                  {n.title}
                </div>
              </button>
            );
          })}

          {/* Hover tooltip */}
          {hovNode && (
            <div style={{
              position:'absolute',
              left: tipLeft, top: tipTop,
              width: 280,
              padding: '10px 12px',
              background: 'var(--bg)',
              border: '1px solid rgba(31,26,20,0.30)',
              borderRadius: 6,
              boxShadow: '0 12px 28px rgba(31,26,20,0.18)',
              zIndex: 10,
              pointerEvents: 'none',
            }}>
              <div style={{
                fontFamily:'var(--mono)', fontSize: 9.5,
                letterSpacing:'0.08em', textTransform:'uppercase',
                color:'var(--ink-soft)', marginBottom: 4,
              }}>
                {hovNode.date}
              </div>
              <div style={{
                fontFamily:'var(--serif)', fontSize: 14, fontWeight: 600,
                color:'var(--ink)', marginBottom: 6, lineHeight: 1.2,
              }}>
                {hovNode.title}
              </div>
              <div style={{
                fontFamily:'var(--serif)', fontSize: 12,
                color:'var(--ink-soft)', lineHeight: 1.4,
              }}>
                {hovNode.note}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { P4_ExperimentTree });
