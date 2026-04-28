// components/primitives.jsx
// Shared visual primitives: GoBoard, PolicyBars, NetGlyph, Caption, ArrowCurve.

// ── GoBoard ─────────────────────────────────────────────────────────────────
// Small rounded-square Go board. `n` = grid size. `stones` = array of
// {x, y, color: 'B'|'W'}. `highlights` = array of {x, y, color?} for
// circled moves. `candidate` = {x, y, color} for a semi-transparent move preview.
function GoBoard({
  n = 3,
  size = 110,
  stones = [],
  highlights = [],
  candidate = null,
  padding,            // computed from size if not provided — ensures corner stones aren't clipped
  radius = 14,
  tint = 1, // 0..1: fades board to bg (used for "imagined" future states)
  style = {},
}) {
  // Stone radius is 0.4 * step; we need padding > stone-radius + a hair.
  // Solve: padding = 0.26 * size keeps corner stones inside the board.
  //   step = (size - 2*padding) / (n - 1)
  //   stone_r = 0.4 * step
  //   we want padding >= stone_r + 2
  // With padding = 0.26*size, step = 0.48*size/(n-1); for n=3 → step=0.24*size,
  //   stone_r = 0.096*size << padding. Safe.
  const defaultPadding = Math.max(12, Math.round(size * 0.22));
  const pad = padding ?? defaultPadding;
  const inner = size - pad * 2;
  const step = inner / (n - 1);
  const board = 'var(--board)';
  const line = 'var(--board-line)';

  const pos = (i) => pad + i * step;

  return (
    <div style={{
      width: size, height: size,
      background: board,
      borderRadius: radius,
      position: 'relative',
      opacity: tint,
      flexShrink: 0,
      boxShadow: '0 1px 0 rgba(0,0,0,0.06), 0 2px 6px rgba(60, 40, 10, 0.08)',
      ...style,
    }}>
      <svg width={size} height={size} style={{ position: 'absolute', inset: 0, display: 'block' }}>
        {/* grid lines */}
        {Array.from({ length: n }).map((_, i) => (
          <React.Fragment key={i}>
            <line x1={pos(0)} y1={pos(i)} x2={pos(n - 1)} y2={pos(i)}
                  stroke={line} strokeWidth={1.2} strokeLinecap="round" />
            <line x1={pos(i)} y1={pos(0)} x2={pos(i)} y2={pos(n - 1)}
                  stroke={line} strokeWidth={1.2} strokeLinecap="round" />
          </React.Fragment>
        ))}
        {/* stones */}
        {stones.map((s, i) => {
          const r = step * 0.4;
          const fill = s.color === 'B' ? 'var(--stone-black)' : 'var(--stone-white)';
          return (
            <g key={`s${i}`}>
              <circle cx={pos(s.x)} cy={pos(s.y)} r={r} fill={fill}
                      stroke={s.color === 'W' ? 'rgba(60,40,20,0.6)' : 'none'} strokeWidth={0.8} />
              {s.color === 'B' && (
                <circle cx={pos(s.x) - r * 0.3} cy={pos(s.y) - r * 0.3} r={r * 0.15}
                        fill="rgba(255,255,255,0.10)" />
              )}
            </g>
          );
        })}
        {/* candidate (semi-transparent preview) */}
        {candidate && (() => {
          const r = step * 0.38;
          const fill = candidate.color === 'B' ? 'var(--stone-black)' : 'var(--stone-white)';
          return (
            <circle cx={pos(candidate.x)} cy={pos(candidate.y)} r={r}
                    fill={fill} opacity={candidate.opacity ?? 0.45}
                    stroke="rgba(60,40,20,0.6)" strokeWidth={0.8}
                    strokeDasharray="2 2" />
          );
        })()}
        {/* highlights (circle around a move) */}
        {highlights.map((h, i) => {
          const r = step * 0.48;
          return (
            <circle key={`h${i}`} cx={pos(h.x)} cy={pos(h.y)} r={r}
                    fill="none" stroke={h.color || 'var(--accent-mcts)'}
                    strokeWidth={2} opacity={h.opacity ?? 1} />
          );
        })}
      </svg>
    </div>
  );
}

// ── PolicyBars ──────────────────────────────────────────────────────────────
// Histogram for the policy distribution. values: array of 0..1.
function PolicyBars({
  values = [],
  width = 110,
  height = 36,
  color = 'var(--ink)',
  gap = 3,
  align = 'bottom',
  style = {},
  baseline = true,
}) {
  const total = values.length;
  const barW = (width - gap * (total - 1)) / total;
  const maxV = Math.max(0.001, ...values);
  return (
    <svg width={width} height={height} style={{ display: 'block', ...style }}>
      {baseline && (
        <line x1={0} y1={height - 0.5} x2={width} y2={height - 0.5}
              stroke="var(--ink-soft)" strokeWidth={0.5} opacity={0.3} />
      )}
      {values.map((v, i) => {
        const h = Math.max(1, (v / maxV) * (height - 2));
        const x = i * (barW + gap);
        const y = height - h;
        return (
          <rect key={i} x={x} y={y} width={barW} height={h}
                fill={color} rx={1} />
        );
      })}
    </svg>
  );
}

// ── NetGlyph ────────────────────────────────────────────────────────────────
// Tiny neural-net glyph (3 layers of dots with connections).
function NetGlyph({ width = 54, height = 30, color = 'var(--ink-soft)', active = false }) {
  const layers = 3;
  const nodes = 3;
  const pad = 3;
  const colW = (width - pad * 2) / (layers - 1);
  const rowH = (height - pad * 2) / (nodes - 1);
  const points = [];
  for (let c = 0; c < layers; c++) {
    for (let r = 0; r < nodes; r++) {
      points.push({ x: pad + c * colW, y: pad + r * rowH, c });
    }
  }
  const edges = [];
  for (let c = 0; c < layers - 1; c++) {
    for (let r1 = 0; r1 < nodes; r1++) {
      for (let r2 = 0; r2 < nodes; r2++) {
        edges.push({
          x1: pad + c * colW, y1: pad + r1 * rowH,
          x2: pad + (c + 1) * colW, y2: pad + r2 * rowH,
        });
      }
    }
  }
  return (
    <svg width={width} height={height} style={{ display: 'block', overflow: 'visible' }}>
      {edges.map((e, i) => (
        <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
              stroke={color} strokeWidth={0.4} opacity={active ? 0.6 : 0.35} />
      ))}
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={1.8}
                fill="var(--bg)" stroke={color} strokeWidth={0.9} />
      ))}
    </svg>
  );
}

// ── Caption ─────────────────────────────────────────────────────────────────
// Small text label — monospace or serif.
function Caption({ children, mono = false, size = 13, color = 'var(--ink-soft)', weight = 400, style = {} }) {
  return (
    <div style={{
      fontFamily: mono ? 'var(--mono)' : 'var(--serif)',
      fontSize: size,
      color,
      fontWeight: weight,
      letterSpacing: mono ? '-0.01em' : '0',
      lineHeight: 1.4,
      ...style,
    }}>
      {children}
    </div>
  );
}

// ── ArrowCurve ──────────────────────────────────────────────────────────────
// Curved SVG edge from (x1,y1) to (x2,y2). Optional arrowhead.
function ArrowCurve({
  x1, y1, x2, y2,
  color = 'var(--ink)',
  strokeWidth = 1.2,
  arrow = false,
  dash = null,
  opacity = 1,
  progress = 1, // 0..1 — partial draw
  curvature = 0.25,
}) {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  // control point: offset perpendicular
  const px = mx - dy * curvature;
  const py = my + dx * curvature;
  const path = `M ${x1} ${y1} Q ${px} ${py} ${x2} ${y2}`;

  // Approximate path length for stroke-dasharray progress
  const pathLen = len * (1 + Math.abs(curvature) * 0.5);
  const dashOffset = (1 - progress) * pathLen;

  const id = React.useId();
  return (
    <g opacity={opacity}>
      {arrow && (
        <defs>
          <marker id={`arr-${id}`} viewBox="0 0 10 10" refX={8} refY={5}
                  markerWidth={5} markerHeight={5} orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill={color} />
          </marker>
        </defs>
      )}
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeDasharray={dash || (progress < 1 ? `${pathLen} ${pathLen}` : undefined)}
        strokeDashoffset={progress < 1 ? dashOffset : undefined}
        markerEnd={arrow && progress > 0.95 ? `url(#arr-${id})` : undefined}
      />
    </g>
  );
}

// ── ValueTag ────────────────────────────────────────────────────────────────
// Monospace "v=0.72" or "N=12" tag.
function ValueTag({ label, value, color = 'var(--ink)', size = 11 }) {
  return (
    <span style={{
      fontFamily: 'var(--mono)',
      fontSize: size,
      color,
      whiteSpace: 'nowrap',
      fontVariantNumeric: 'tabular-nums',
    }}>
      {label}<span style={{ opacity: 0.55 }}>=</span>{value}
    </span>
  );
}

// ── SectionLabel ────────────────────────────────────────────────────────────
// Real <h2> so screen readers can navigate the deck by headings. Visual
// styling is unchanged — we override default browser margins here.
function SectionLabel({ num, title, color = 'var(--ink)' }) {
  return (
    <h2 style={{
      display: 'flex', alignItems: 'baseline', gap: 10,
      margin: 0, padding: 0, fontWeight: 500,
    }}>
      <span style={{
        fontFamily: 'var(--mono)',
        fontSize: 12,
        color: 'var(--ink-soft)',
        letterSpacing: '0.08em',
      }} aria-hidden="true">{num}</span>
      <span style={{
        fontFamily: 'var(--serif)',
        fontSize: 22,
        fontWeight: 500,
        color,
        letterSpacing: '-0.01em',
      }}>{title}</span>
    </h2>
  );
}

Object.assign(window, {
  GoBoard, PolicyBars, NetGlyph, Caption, ArrowCurve, ValueTag, SectionLabel,
});
