// components/p4_variable_board.jsx
// Part 4 · slide 11 — variable board sizing.
// Adapted from the Claude Design "Neural Net Viz.html" handoff: an
// orthographic forward-pass diagram of the size-invariant Go ResNet,
// with a 9×9 board padded into 19×19 + mask. The design's H1
// ("One-hot input → masked residual tower → policy & value heads") is
// dropped; the slide instead uses our standard SlideHeader and a
// warm-start callout below the diagram.

const NN_COLORS = {
  bg: "#f6f3ec",
  ink: "#1a1a1a",
  faint: "#cdc4ae",
  rule: "#7a7460",
  grid: "#5a5544",
  mask: "#d8cfb6",
  maskHatch: "#8a8268",
  empty: "#d4c8a3",
  self: "#2c3d75",
  opp:  "#a04a25",
  feat: "#1a1a1a",
  selfSoft: "rgba(44, 61, 117, 0.32)",
  oppSoft:  "rgba(160, 74, 37, 0.32)",
  block: "#efe9d6",
  blockEdge: "#9c9276",
  blockSoft: "#f4eedc",
};

const NN_BOARD_9 = [
  [0,0,0,0,0,0,0,0,0],
  [0,0,2,0,0,0,0,0,0],
  [0,2,1,2,0,0,0,1,0],
  [0,0,1,1,2,0,1,2,0],
  [0,0,0,1,2,2,1,2,0],
  [0,0,0,0,1,1,2,0,0],
  [0,0,0,0,0,0,0,0,0],
  [0,0,1,0,0,0,2,1,0],
  [0,0,0,0,0,0,0,0,0],
];

function nnPadBoard(src, padN) {
  const N = src.length;
  const out = Array.from({length: padN}, () => Array(padN).fill(0));
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) out[r][c] = src[r][c];
  return out;
}
function nnPadMask(N, padN) {
  const m = Array.from({length: padN}, () => Array(padN).fill(0));
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) m[r][c] = 1;
  return m;
}

function NN_ShapePill({x, y, text, kind = "default"}) {
  const fill = kind === "accent" ? NN_COLORS.self : NN_COLORS.block;
  const textFill = kind === "accent" ? "#fff" : NN_COLORS.ink;
  const width = Math.max(72, text.length * 6.4 + 14);
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect x={0} y={0} width={width} height={20} rx={3}
            fill={fill} stroke={NN_COLORS.blockEdge} strokeWidth={0.7}/>
      <text x={width/2} y={14} textAnchor="middle"
            fontFamily="JetBrains Mono, monospace" fontSize="10"
            fill={textFill}>{text}</text>
    </g>
  );
}

function NN_PaddedBoard({N9, N19, cell, board, mask, opacity = 1}) {
  return (
    <g opacity={opacity}>
      <g transform="matrix(1, 0, 0.55, 0.85, 0, 0)">
        <rect x={0} y={0} width={N19*cell} height={N19*cell}
              fill={NN_COLORS.empty} stroke={NN_COLORS.rule} strokeWidth={0.8}/>
        <rect x={N9*cell} y={0} width={(N19-N9)*cell} height={N19*cell}
              fill="url(#nnHatch)" opacity={0.7}/>
        <rect x={0} y={N9*cell} width={N9*cell} height={(N19-N9)*cell}
              fill="url(#nnHatch)" opacity={0.7}/>
        {Array.from({length: N19+1}, (_, i) => (
          <g key={i}>
            <line x1={0} y1={i*cell} x2={N19*cell} y2={i*cell}
                  stroke={NN_COLORS.grid} strokeWidth={0.4} opacity={0.6}/>
            <line x1={i*cell} y1={0} x2={i*cell} y2={N19*cell}
                  stroke={NN_COLORS.grid} strokeWidth={0.4} opacity={0.6}/>
          </g>
        ))}
        {board.flatMap((row, r) => row.map((v, c) => {
          if (mask[r][c] === 0 || v === 0) return null;
          const cx = c*cell + cell/2, cy = r*cell + cell/2;
          return <circle key={`s${r}-${c}`} cx={cx} cy={cy} r={cell*0.36}
                         fill={v === 1 ? NN_COLORS.self : NN_COLORS.opp}/>;
        }))}
        <rect x={0} y={0} width={N9*cell} height={N9*cell}
              fill="none" stroke={NN_COLORS.self} strokeWidth={1.4} opacity={0.9}/>
      </g>
    </g>
  );
}

function NN_ChannelPlane({N, cell, board, mask, channel, opacity = 1}) {
  const W = N * cell;
  const fill = channel === 0 ? NN_COLORS.empty : channel === 1 ? NN_COLORS.selfSoft : NN_COLORS.oppSoft;
  const dotFill = channel === 0 ? "#8a8268" : channel === 1 ? NN_COLORS.self : NN_COLORS.opp;
  return (
    <g opacity={opacity}>
      <g transform="matrix(1, 0, 0.55, 0.85, 0, 0)">
        <rect x={0} y={0} width={W} height={W} fill={fill} stroke={NN_COLORS.rule} strokeWidth={0.6}/>
        {board.flatMap((row, r) => row.map((v, c) => {
          if (channel === 0 && mask && mask[r][c] === 0) return null;
          if (v !== channel) return null;
          const cx = c * cell + cell/2;
          const cy = r * cell + cell/2;
          return <rect key={`d${r}-${c}`} x={cx-cell*0.34} y={cy-cell*0.34}
                       width={cell*0.68} height={cell*0.68} fill={dotFill}/>;
        }))}
        {mask && mask.flatMap((row, r) => row.map((m, c) => m === 0 ? (
          <rect key={`mm${r}-${c}`} x={c*cell} y={r*cell} width={cell} height={cell}
                fill="url(#nnHatch)" opacity={0.7} />
        ) : null))}
      </g>
    </g>
  );
}

function NN_FeatureMap({N, cell, mask, seed = 1, opacity = 1, intensity = 1}) {
  const cells = [];
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const v = ((Math.sin((r+1)*12.9898 + (c+1)*78.233 + seed*3.7) * 43758.5453) % 1 + 1) % 1;
      const inMask = !mask || mask[r][c] === 1;
      const a = inMask ? (0.15 + v * 0.7) * intensity : 0;
      cells.push(<rect key={`f${r}-${c}`} x={c*cell} y={r*cell} width={cell} height={cell}
              fill={NN_COLORS.feat} opacity={a}/>);
    }
  }
  return (
    <g opacity={opacity}>
      <g transform="matrix(1, 0, 0.55, 0.85, 0, 0)">
        <rect x={0} y={0} width={N*cell} height={N*cell} fill={NN_COLORS.bg} stroke={NN_COLORS.rule} strokeWidth={0.5}/>
        {cells}
        {mask && mask.flatMap((row, r) => row.map((m, c) => m === 0 ? (
          <rect key={`fm${r}-${c}`} x={c*cell} y={r*cell} width={cell} height={cell}
                fill={NN_COLORS.mask} opacity={0.94} />
        ) : null))}
      </g>
    </g>
  );
}

function NN_FeatureStack({N, cell, mask, nSlices = 4, baseSeed = 1, opacity = 1, intensity = 1}) {
  return (
    <g opacity={opacity}>
      {Array.from({length: nSlices}, (_, i) => {
        const off = (nSlices - 1 - i) * 4;
        return (
          <g key={i} transform={`translate(${off * 0.7}, ${-off * 0.5})`}>
            <NN_FeatureMap N={N} cell={cell} mask={mask} seed={baseSeed + i*7}
                        intensity={intensity * (0.6 + i*0.1)}/>
          </g>
        );
      })}
    </g>
  );
}

function NN_Arrow({x1, y1, x2, y2, dashed = false, stroke = NN_COLORS.ink, strokeWidth = 1.2,
                label, labelPos = "mid", opacity = 1}) {
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx*dx + dy*dy);
  const ux = dx/len, uy = dy/len;
  const tx = x2 - ux * 2, ty = y2 - uy * 2;
  return (
    <g opacity={opacity}>
      <line x1={x1} y1={y1} x2={tx} y2={ty} stroke={stroke} strokeWidth={strokeWidth}
            strokeDasharray={dashed ? "4 4" : undefined} opacity={0.75}/>
      <polygon points={`${x2},${y2} ${x2 - ux*9 - uy*4},${y2 - uy*9 + ux*4} ${x2 - ux*9 + uy*4},${y2 - uy*9 - ux*4}`}
               fill={stroke} opacity={0.85}/>
      {label && (
        <text x={(x1+x2)/2} y={(y1+y2)/2 + (labelPos === "below" ? 14 : -6)}
              textAnchor="middle"
              fontFamily="JetBrains Mono, monospace" fontSize="9.5"
              fill="#4a4536" letterSpacing="0.06em">{label}</text>
      )}
    </g>
  );
}

function NN_MaskedResBlockDetail({x, y, opacity = 1, highlight = 0}) {
  const ops = [
    {label: "conv 3×3 · 128→128", h: 22},
    {label: "MaskedBN",            h: 18},
    {label: "ReLU",                h: 18},
    {label: "conv 3×3 · 128→128",  h: 22},
    {label: "MaskedBN",            h: 18},
  ];
  const W = 200;
  let cy = 0;
  const pos = ops.map(o => { const p = cy; cy += o.h + 8; return p; });
  const totalH = cy + 38;
  return (
    <g transform={`translate(${x}, ${y})`} opacity={opacity}>
      <rect x={-12} y={-32} width={W + 24} height={totalH + 60} rx={8}
            fill={NN_COLORS.blockSoft} stroke={NN_COLORS.blockEdge} strokeWidth={1}/>
      <text x={-6} y={-40} fontFamily="JetBrains Mono, monospace" fontSize="9.5"
            fill="#7a7460" letterSpacing="0.18em">MASKED RES BLOCK · DETAIL</text>
      <text x={W/2} y={-14} textAnchor="middle"
            fontFamily="JetBrains Mono, monospace" fontSize="9"
            fill="#7a7460">in : (B, 128, H, W)</text>
      {ops.map((o, i) => {
        const active = highlight > i / ops.length;
        return (
          <g key={i} transform={`translate(0, ${pos[i]})`}>
            <rect x={0} y={0} width={W} height={o.h} rx={4}
                  fill={active ? NN_COLORS.self : NN_COLORS.bg}
                  stroke={NN_COLORS.blockEdge} strokeWidth={0.8}/>
            <text x={W/2} y={o.h/2 + 4} textAnchor="middle"
                  fontFamily="JetBrains Mono, monospace" fontSize="10"
                  fill={active ? "#fff" : NN_COLORS.ink}>{o.label}</text>
            {i < ops.length - 1 && (
              <line x1={W/2} y1={o.h} x2={W/2} y2={o.h + 8}
                    stroke={NN_COLORS.blockEdge} strokeWidth={0.7}/>
            )}
            <text x={W + 6} y={o.h/2 + 4}
                  fontFamily="JetBrains Mono, monospace" fontSize="8.5"
                  fill={NN_COLORS.opp}>× mask</text>
          </g>
        );
      })}
      <path d={`M -8 -6 L -8 ${cy + 18} L ${W/2 - 10} ${cy + 18}`}
            fill="none" stroke={NN_COLORS.opp} strokeWidth={1.4} opacity={0.85}/>
      <text x={-2} y={cy/2}
            fontFamily="JetBrains Mono, monospace" fontSize="9"
            fill={NN_COLORS.opp} transform={`rotate(-90, -2, ${cy/2})`}>skip</text>
      <g transform={`translate(${W/2}, ${cy + 18})`}>
        <line x1={0} y1={-8} x2={0} y2={-1} stroke={NN_COLORS.blockEdge} strokeWidth={0.7}/>
        <circle cx={0} cy={6} r={9} fill={NN_COLORS.bg} stroke={NN_COLORS.blockEdge} strokeWidth={1}/>
        <text x={0} y={9.5} textAnchor="middle"
              fontFamily="JetBrains Mono, monospace" fontSize="11" fill={NN_COLORS.ink}>+</text>
        <line x1={0} y1={15} x2={0} y2={22} stroke={NN_COLORS.blockEdge} strokeWidth={0.7}/>
      </g>
      <g transform={`translate(0, ${cy + 40})`}>
        <rect x={0} y={0} width={W} height={20} rx={4}
              fill={NN_COLORS.bg} stroke={NN_COLORS.blockEdge} strokeWidth={0.8}/>
        <text x={W/2} y={14} textAnchor="middle"
              fontFamily="JetBrains Mono, monospace" fontSize="10"
              fill={NN_COLORS.ink}>ReLU</text>
      </g>
      <text x={W/2} y={cy + 76} textAnchor="middle"
            fontFamily="JetBrains Mono, monospace" fontSize="9"
            fill="#7a7460">out : (B, 128, H, W)</text>
    </g>
  );
}

function NN_GoNetSceneInner({t}) {
  const N9 = 9, N19 = 19;
  const board9 = NN_BOARD_9;
  const board19 = React.useMemo(() => nnPadBoard(board9, N19), []);
  const mask19  = React.useMemo(() => nnPadMask(N9, N19), []);

  const pBoard = clamp((t - 0.0) / 1.4, 0, 1);
  const pMorph = clamp((t - 1.4) / 1.2, 0, 1);
  const p2     = clamp((t - 2.6) / 1.6, 0, 1);
  const p3     = clamp((t - 4.2) / 1.8, 0, 1);
  const p4     = clamp((t - 6.0) / 1.6, 0, 1);

  const inputOp = Easing.easeOutCubic(clamp(t / 1.0, 0, 1));
  const morph   = Easing.easeOutCubic(pMorph);
  const stemOp  = Easing.easeOutCubic(p2);
  const towerOp = Easing.easeOutCubic(p3);
  const headsOp = Easing.easeOutCubic(p4);
  const detailOp = Easing.easeOutCubic(clamp((t - 2.4) / 1.2, 0, 1));

  const blockHighlight = (Math.floor(t * 1.5) % 10);
  const pulse = 0.85 + 0.15 * Math.sin(t * 2.4);
  const valueP = 0.5 + 0.32 * Math.sin(t * 0.9 + 1.2);

  return (
    <g>
      <defs>
        <pattern id="nnHatch" patternUnits="userSpaceOnUse" width="6" height="6"
                 patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke={NN_COLORS.maskHatch} strokeWidth="1.4"/>
        </pattern>
      </defs>

      {/* Title block from the design is intentionally omitted —
          our slide uses the page's SlideHeader instead. */}

      {/* INPUT (padded → one-hot morph) */}
      <g transform="translate(60, 170)" opacity={inputOp}>
        <text x={0} y={-12} fontFamily="JetBrains Mono, monospace" fontSize="10"
              fill="#7a7460" letterSpacing="0.18em" opacity={1 - morph}>
          INPUT · PADDED 19×19
        </text>
        <text x={0} y={-12} fontFamily="JetBrains Mono, monospace" fontSize="10"
              fill="#7a7460" letterSpacing="0.18em" opacity={morph}>
          INPUT · ONE-HOT
        </text>
        <g opacity={(1 - morph) * Easing.easeOutCubic(pBoard)}>
          <NN_PaddedBoard N9={N9} N19={N19} cell={9} board={board19} mask={mask19}/>
        </g>
        <g opacity={morph}>
          {[2, 1, 0].map((ch, idx) => {
            const cell = 9;
            const off = idx * 22 * morph;
            const slide = idx * 6 * morph;
            const labels = {0: "ch 0 · empty", 1: "ch 1 · self", 2: "ch 2 · opp"};
            return (
              <g key={ch} transform={`translate(${slide}, ${-off})`}>
                <NN_ChannelPlane N={N19} cell={cell} board={board19} mask={mask19} channel={ch}/>
                <text x={N19*cell + 14} y={N19*cell*0.85 - 8}
                      fontFamily="JetBrains Mono, monospace" fontSize="9"
                      fill="#4a4536" opacity={clamp(morph*1.4 - 0.3, 0, 1)}>{labels[ch]}</text>
              </g>
            );
          })}
        </g>
        <NN_ShapePill x={0} y={N19*9*0.85 + 22} text="(B, 3, 19, 19)" kind="accent"/>
        <text x={0} y={N19*9*0.85 + 60} fontFamily="JetBrains Mono, monospace"
              fontSize="9" fill="#7a7460">9×9 board · upper-left · mask=1 inside</text>
      </g>

      <NN_Arrow x1={272} y1={228} x2={346} y2={228} dashed={true}
             label="× mask" labelPos="below" opacity={inputOp}/>

      {/* STEM CONV */}
      <g transform="translate(348, 196)" opacity={inputOp}>
        <text x={0} y={-12} fontFamily="JetBrains Mono, monospace" fontSize="10"
              fill="#7a7460" letterSpacing="0.18em">STEM</text>
        <rect x={0} y={0} width={106} height={64} rx={5}
              fill={NN_COLORS.blockSoft} stroke={NN_COLORS.blockEdge} strokeWidth={1}/>
        <text x={53} y={22} textAnchor="middle"
              fontFamily="JetBrains Mono, monospace" fontSize="11"
              fill={NN_COLORS.ink}>input_conv</text>
        <text x={53} y={36} textAnchor="middle"
              fontFamily="JetBrains Mono, monospace" fontSize="9"
              fill="#7a7460">3×3 · 3 → 128</text>
        <text x={53} y={50} textAnchor="middle"
              fontFamily="JetBrains Mono, monospace" fontSize="9"
              fill="#7a7460">+ MaskedBN + ReLU</text>
        <NN_ShapePill x={-8} y={86} text="(B, 128, 19, 19)" kind="accent"/>
      </g>

      <NN_Arrow x1={458} y1={228} x2={492} y2={228} dashed={true} opacity={stemOp}/>

      {/* TOWER */}
      <g transform="translate(494, 170)" opacity={stemOp}>
        <text x={0} y={-12} fontFamily="JetBrains Mono, monospace" fontSize="10"
              fill="#7a7460" letterSpacing="0.18em">TOWER · 10 × MASKED RES BLOCK</text>
        <rect x={-6} y={0} width={460} height={130} rx={6}
              fill="none" stroke={NN_COLORS.blockEdge} strokeWidth={0.8}
              strokeDasharray="4 4" opacity={0.5}/>
        {Array.from({length: 10}, (_, i) => {
          const col = i % 5;
          const row = Math.floor(i / 5);
          const x = col * 88;
          const y = row * 50;
          const blockProgress = clamp(towerOp * 10 - i * 0.6, 0, 1);
          const isActive = blockHighlight === i && t > 4.0;
          return (
            <g key={i} transform={`translate(${x + 6}, ${y + 16})`} opacity={blockProgress}>
              <rect x={0} y={0} width={80} height={36} rx={4}
                    fill={isActive ? NN_COLORS.self : NN_COLORS.blockSoft}
                    stroke={NN_COLORS.blockEdge} strokeWidth={0.9}/>
              <text x={40} y={15} textAnchor="middle"
                    fontFamily="JetBrains Mono, monospace" fontSize="9"
                    fill={isActive ? "#fff" : NN_COLORS.ink}>block {i+1}</text>
              <text x={40} y={26} textAnchor="middle"
                    fontFamily="JetBrains Mono, monospace" fontSize="8"
                    fill={isActive ? "#e8e2cc" : "#7a7460"}>128 ch · masked</text>
              {col < 4 && (
                <line x1={80} y1={18} x2={88} y2={18}
                      stroke={NN_COLORS.blockEdge} strokeWidth={0.7}/>
              )}
              {i === 4 && (
                <path d={`M 80 18 Q 88 18 88 26 L 88 60 Q 88 66 80 66 L -416 66 Q -424 66 -424 74 L -424 80`}
                      fill="none" stroke={NN_COLORS.blockEdge} strokeWidth={0.7} opacity={0.5}
                      strokeDasharray="3 3"/>
              )}
            </g>
          );
        })}
        <NN_ShapePill x={0} y={140} text="(B, 128, 19, 19)" kind="accent"/>
        <text x={120} y={154} fontFamily="JetBrains Mono, monospace" fontSize="9"
              fill="#7a7460">spatial size preserved · mask preserved</text>
      </g>

      <NN_Arrow x1={920} y1={236} x2={956} y2={236} dashed={true} opacity={towerOp}/>

      {/* FINAL FEATURES */}
      <g transform="translate(958, 168)" opacity={towerOp}>
        <text x={0} y={-12} fontFamily="JetBrains Mono, monospace" fontSize="10"
              fill="#7a7460" letterSpacing="0.18em">FINAL FEATURES</text>
        <NN_FeatureStack N={N19} cell={6} mask={mask19} nSlices={5} baseSeed={42}
                      intensity={pulse}/>
        <NN_ShapePill x={0} y={N19*6*0.85 + 24} text="(B, 128, 19, 19)" kind="accent"/>
        <text x={0} y={N19*6*0.85 + 60} fontFamily="JetBrains Mono, monospace"
              fontSize="9" fill="#7a7460">128 spatial channels</text>
      </g>

      {/* Heads */}
      <g opacity={headsOp}>
        <NN_Arrow x1={1130} y1={210} x2={1188} y2={140} dashed={true}
               label="1×1 conv" labelPos="above"/>
        <NN_Arrow x1={1130} y1={260} x2={1188} y2={350} dashed={true}/>
        <text x={1080} y={310}
              fontFamily="JetBrains Mono, monospace" fontSize="9.5"
              fill="#4a4536" letterSpacing="0.06em">masked-avg-pool</text>
      </g>

      {/* POLICY HEAD */}
      <g transform="translate(1190, 90)" opacity={headsOp}>
        <text x={0} y={-12} fontFamily="JetBrains Mono, monospace" fontSize="10"
              fill="#7a7460" letterSpacing="0.18em">POLICY HEAD</text>
        <g>
          <rect x={0} y={0} width={88} height={26} rx={4}
                fill={NN_COLORS.blockSoft} stroke={NN_COLORS.blockEdge} strokeWidth={0.9}/>
          <text x={44} y={12} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="10" fill={NN_COLORS.ink}>conv 1×1</text>
          <text x={44} y={22} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#7a7460">128 → 1</text>
          <line x1={88} y1={13} x2={102} y2={13} stroke={NN_COLORS.blockEdge} strokeWidth={0.7}/>
          <rect x={102} y={0} width={108} height={26} rx={4}
                fill={NN_COLORS.blockSoft} stroke={NN_COLORS.blockEdge} strokeWidth={0.9}/>
          <text x={156} y={12} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="10" fill={NN_COLORS.ink}>flatten · mask</text>
          <text x={156} y={22} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill={NN_COLORS.opp}>pad → −∞</text>
          <line x1={210} y1={13} x2={224} y2={13} stroke={NN_COLORS.blockEdge} strokeWidth={0.7}/>
          <rect x={224} y={0} width={68} height={26} rx={4}
                fill={NN_COLORS.blockSoft} stroke={NN_COLORS.blockEdge} strokeWidth={0.9}/>
          <text x={258} y={12} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="10" fill={NN_COLORS.ink}>+ pass_fc</text>
          <text x={258} y={22} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#7a7460">pooled → 1</text>
        </g>
        <NN_Arrow x1={146} y1={36} x2={146} y2={58} dashed={false}/>
        <text x={155} y={50} fontFamily="JetBrains Mono, monospace" fontSize="9"
              fill="#4a4536">softmax</text>
        <g transform="translate(0, 70)">
          <g transform="matrix(1, 0, 0.55, 0.85, 0, 0)">
            <rect x={0} y={0} width={N19*6} height={N19*6} fill={NN_COLORS.bg}
                  stroke={NN_COLORS.rule} strokeWidth={0.5}/>
            {Array.from({length: N19}, (_, r) =>
              Array.from({length: N19}, (_, c) => {
                const inActive = r < N9 && c < N9;
                if (!inActive) {
                  return <rect key={`p${r}-${c}`} x={c*6} y={r*6} width={6} height={6}
                               fill="url(#nnHatch)" opacity={0.7}/>;
                }
                const hot = (r === 4 && c === 5) ? 1.0
                  : (r === 6 && c === 5) ? 0.7
                  : (r === 2 && c === 6) ? 0.55
                  : (r === 5 && c === 7) ? 0.4
                  : 0;
                return (
                  <rect key={`p${r}-${c}`} x={c*6} y={r*6} width={6} height={6}
                        fill={NN_COLORS.self} opacity={hot * 0.9}/>
                );
              })
            )}
            {Array.from({length: N19+1}, (_, i) => (
              <g key={i}>
                <line x1={0} y1={i*6} x2={N19*6} y2={i*6}
                      stroke={NN_COLORS.grid} strokeWidth={0.3} opacity={0.5}/>
                <line x1={i*6} y1={0} x2={i*6} y2={N19*6}
                      stroke={NN_COLORS.grid} strokeWidth={0.3} opacity={0.5}/>
              </g>
            ))}
          </g>
          <text x={185} y={28} fontFamily="JetBrains Mono, monospace" fontSize="10"
                fill="#3a3527">π(a | s)</text>
          <text x={185} y={48} fontFamily="JetBrains Mono, monospace" fontSize="9"
                fill={NN_COLORS.opp}>argmax · (4, 5)</text>
          <text x={185} y={66} fontFamily="JetBrains Mono, monospace" fontSize="9"
                fill="#7a7460">+ pass action</text>
          <NN_ShapePill x={0} y={114} text="(B, 19·19 + 1)" kind="accent"/>
        </g>
      </g>

      {/* VALUE HEAD */}
      <g transform="translate(1190, 350)" opacity={headsOp}>
        <text x={0} y={-12} fontFamily="JetBrains Mono, monospace" fontSize="10"
              fill="#7a7460" letterSpacing="0.18em">VALUE HEAD</text>
        <text x={0} y={2} fontFamily="JetBrains Mono, monospace" fontSize="9"
              fill="#7a7460">pooled</text>
        <NN_ShapePill x={42} y={-8} text="(B, 128)"/>
        <g transform="translate(0, 26)">
          <rect x={0} y={0} width={66} height={26} rx={4}
                fill={NN_COLORS.blockSoft} stroke={NN_COLORS.blockEdge} strokeWidth={0.9}/>
          <text x={33} y={12} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="10" fill={NN_COLORS.ink}>FC</text>
          <text x={33} y={22} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#7a7460">128 → 64</text>
          <line x1={66} y1={13} x2={80} y2={13} stroke={NN_COLORS.blockEdge} strokeWidth={0.7}/>
          <rect x={80} y={0} width={48} height={26} rx={4}
                fill={NN_COLORS.blockSoft} stroke={NN_COLORS.blockEdge} strokeWidth={0.9}/>
          <text x={104} y={16} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="10" fill={NN_COLORS.ink}>ReLU</text>
          <line x1={128} y1={13} x2={142} y2={13} stroke={NN_COLORS.blockEdge} strokeWidth={0.7}/>
          <rect x={142} y={0} width={66} height={26} rx={4}
                fill={NN_COLORS.blockSoft} stroke={NN_COLORS.blockEdge} strokeWidth={0.9}/>
          <text x={175} y={12} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="10" fill={NN_COLORS.ink}>FC</text>
          <text x={175} y={22} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#7a7460">64 → 1</text>
        </g>
        <NN_ShapePill x={0} y={62} text="v : (B,)" kind="accent"/>
        <text x={70} y={76} fontFamily="JetBrains Mono, monospace" fontSize="9"
              fill="#7a7460">→ σ(v)</text>
        <g transform="translate(0, 96)">
          <text x={0} y={-3} fontFamily="JetBrains Mono, monospace" fontSize="9"
                fill="#7a7460">P(self wins)</text>
          <rect x={0} y={4} width={208} height={14} rx={2} fill={NN_COLORS.faint}/>
          <rect x={0} y={4} width={208 * valueP} height={14} rx={2}
                fill={NN_COLORS.self} opacity={0.9}/>
          <line x1={104} y1={1} x2={104} y2={21} stroke={NN_COLORS.ink} strokeWidth={0.6} opacity={0.4}/>
          <text x={0} y={32} fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#7a7460">0</text>
          <text x={104} y={32} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#7a7460">0.5</text>
          <text x={208} y={32} textAnchor="end"
                fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#7a7460">1</text>
          <text x={208 * valueP} y={-2} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize="9.5" fill={NN_COLORS.self}
                fontWeight="600">{valueP.toFixed(2)}</text>
        </g>
      </g>

      {/* SIDE PANEL — MaskedResBlock detail */}
      <g transform="translate(60, 410)" opacity={detailOp}>
        <text x={0} y={-4} fontFamily="JetBrains Mono, monospace" fontSize="11"
              letterSpacing="0.22em" fill={NN_COLORS.ink}>WHAT'S INSIDE EACH BLOCK</text>
        <text x={0} y={12} fontFamily="JetBrains Mono, monospace" fontSize="9"
              fill="#7a7460">Two masked 3×3 convs around an additive skip;</text>
        <text x={0} y={26} fontFamily="JetBrains Mono, monospace" fontSize="9"
              fill="#7a7460">mask multiplied at every op.</text>
      </g>
      <NN_MaskedResBlockDetail x={92} y={478} opacity={detailOp}
        highlight={clamp(((t * 0.55) % 1.0) * 1.4, 0, 1.0)}/>
      <g opacity={detailOp * 0.5}>
        <path d={`M 480 220 Q 350 340 200 470`}
              fill="none" stroke={NN_COLORS.blockEdge} strokeWidth={0.7}
              strokeDasharray="2 4"/>
      </g>

      {/* Footer legend */}
      <g transform="translate(440, 695)">
        <g>
          <rect x={0} y={-9} width={12} height={12} fill={NN_COLORS.self} rx={1}/>
          <text x={20} y={1} fontFamily="JetBrains Mono, monospace" fontSize="11"
                fill="#3a3527">self · ch 1</text>
        </g>
        <g transform="translate(140, 0)">
          <rect x={0} y={-9} width={12} height={12} fill={NN_COLORS.opp} rx={1}/>
          <text x={20} y={1} fontFamily="JetBrains Mono, monospace" fontSize="11"
                fill="#3a3527">opp · ch 2</text>
        </g>
        <g transform="translate(280, 0)">
          <rect x={0} y={-9} width={12} height={12} fill={NN_COLORS.empty}
                stroke={NN_COLORS.grid} strokeWidth={0.6} rx={1}/>
          <text x={20} y={1} fontFamily="JetBrains Mono, monospace" fontSize="11"
                fill="#3a3527">empty · ch 0</text>
        </g>
        <g transform="translate(440, 0)">
          <rect x={0} y={-9} width={12} height={12} fill="url(#nnHatch)" rx={1}
                stroke={NN_COLORS.rule} strokeWidth={0.5}/>
          <text x={20} y={1} fontFamily="JetBrains Mono, monospace" fontSize="11"
                fill="#3a3527">mask=0 · padded</text>
        </g>
        <g transform="translate(600, 0)">
          <line x1={0} y1={-3} x2={14} y2={-3} stroke={NN_COLORS.opp} strokeWidth={1.6}/>
          <text x={20} y={1} fontFamily="JetBrains Mono, monospace" fontSize="11"
                fill="#3a3527">residual / skip</text>
        </g>
        <g transform="translate(760, 0)">
          <rect x={0} y={-9} width={20} height={12} rx={2} fill={NN_COLORS.self}/>
          <text x={28} y={1} fontFamily="JetBrains Mono, monospace" fontSize="11"
                fill="#3a3527">tensor shape</text>
        </g>
      </g>
    </g>
  );
}

// ── Slide wrapper ──────────────────────────────────────────────────────────
function P4_VariableBoard() {
  const { localTime: lt } = useSprite();

  return (
    <>
      <SlideHeader num="13" title="Variable board sizing" maxWidth={820} subtitle={<>
        I found it helpful to train a strong model on 9x9 first, and then use the data collected to warm-start a 19x19 model. To have the model handle both 9×9 and 19×19 inputs, we use the Fully Convolutional ResNet with masking from KataGo. Stones are pinned to the upper-left of a padded 19×19 canvas, and a binary mask multiplies through every conv, BN, and head.
      </>} />

      <div style={{
        position: 'absolute',
        left: 170, right: 20, top: 168, bottom: 36,
      }}>
        <svg viewBox="0 130 1600 580"
             width="100%" height="100%"
             preserveAspectRatio="xMidYMid meet"
             style={{ display: 'block' }}>
          <NN_GoNetSceneInner t={lt} />
        </svg>
      </div>
    </>
  );
}

Object.assign(window, { P4_VariableBoard });
