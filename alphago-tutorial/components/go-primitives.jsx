// components/go-primitives.jsx
// A custom Go animation library for Part 1 (Rules of Go) slides.
//
// Core component: <GoAnim> — a Go board that plays a timeline of stone
// placements, captures, and overlay annotations as a function of `lt`
// (the local sprite time, in seconds).
//
// Example:
//   <GoAnim n={5} size={400} lt={lt}
//     moves={[
//       { at: 0.3, x: 2, y: 2, color: 'W' },
//       { at: 1.0, x: 2, y: 1, color: 'B', number: 1 },
//     ]}
//     captures={[
//       { at: 5.0, x: 2, y: 2, fadeOut: 0.4 },
//     ]}
//     annotations={[
//       { at: 1.2, type: 'liberty', x: 1, y: 2 },
//       { at: 4.5, until: 5.0, type: 'atari', x: 2, y: 2 },
//     ]}
//   />
//
// Supported annotation types:
//   'eye'        — dashed ring at an empty point
//   'liberty'    — solid dot at an empty adjacent point
//   'atari'      — ring around a stone in atari
//   'ko'         — square indicator at the ko point
//   'ban'        — red X (suicide / ko forbidden)
//   'territory'  — tinted square over empty point(s); props: {points?, owner}
//   'dame'       — neutral-point dot
//   'label'      — text drawn at an intersection
//   'tip'        — small text above an intersection (e.g. "atari")
//   'arrow'      — arrow from one intersection to another
//   'circle'     — generic highlight ring
//   'group'      — translucent halo around a group of points

function goLayout(n, size, padding) {
  const pad = padding ?? Math.max(12, Math.round(size * 0.22));
  const inner = size - pad * 2;
  const step = n > 1 ? inner / (n - 1) : inner;
  const pos = (i) => pad + i * step;
  const stoneR = step * 0.4;
  return { pad, inner, step, pos, stoneR, size };
}

function GoAnim({
  n = 5, size = 400, lt = 0,
  moves = [], captures = [], annotations = [],
  style = {},
  radius,
}) {
  const L = goLayout(n, size);
  const corner = radius ?? Math.round(size * 0.035);

  const computedStones = moves.map((m) => {
    const cap = captures.find(c =>
      c.x === m.x && c.y === m.y && c.at >= m.at
    );
    const placed = lt >= m.at;
    if (!placed) return null;
    const fadeIn = m.fadeIn ?? 0.25;
    const inT = Easing.easeOutCubic(clamp((lt - m.at) / fadeIn, 0, 1));

    let outT = 1;
    let captured = false;
    if (cap && lt >= cap.at) {
      captured = true;
      const fadeOut = cap.fadeOut ?? 0.35;
      outT = 1 - Easing.easeInCubic(clamp((lt - cap.at) / fadeOut, 0, 1));
    }

    const opacity = inT * outT;
    if (opacity < 0.01) return null;
    const scale = 0.55 + 0.45 * Math.min(1, inT);
    return { ...m, opacity, scale, captured };
  }).filter(Boolean);

  const below = annotations.filter(a => a.layer === 'below' || a.type === 'territory' || a.type === 'group');
  const above = annotations.filter(a => !(a.layer === 'below' || a.type === 'territory' || a.type === 'group'));

  return (
    <div style={{
      width: size, height: size,
      background: 'var(--board)',
      borderRadius: corner,
      position: 'relative',
      flexShrink: 0,
      boxShadow: '0 1px 0 rgba(0,0,0,0.06), 0 2px 6px rgba(60, 40, 10, 0.08)',
      ...style,
    }}>
      <svg width={size} height={size} style={{ position: 'absolute', inset: 0, display: 'block' }}>
        {/* grid */}
        {Array.from({ length: n }).map((_, i) => (
          <React.Fragment key={i}>
            <line x1={L.pos(0)} y1={L.pos(i)} x2={L.pos(n - 1)} y2={L.pos(i)}
                  stroke="var(--board-line)" strokeWidth={1.2} strokeLinecap="round" />
            <line x1={L.pos(i)} y1={L.pos(0)} x2={L.pos(i)} y2={L.pos(n - 1)}
                  stroke="var(--board-line)" strokeWidth={1.2} strokeLinecap="round" />
          </React.Fragment>
        ))}

        {/* below-stone annotations */}
        {below.map((a, i) => (
          <GoAnnot key={`b-${i}`} a={a} lt={lt} L={L} n={n} />
        ))}

        {/* stones */}
        {computedStones.map((s, i) => {
          const fill = s.color === 'B' ? 'var(--stone-black)' : 'var(--stone-white)';
          const strokeC = s.color === 'W' ? 'rgba(60,40,20,0.65)' : 'none';
          return (
            <g key={`s-${s.x}-${s.y}-${i}`} opacity={s.opacity}
               transform={`translate(${L.pos(s.x)} ${L.pos(s.y)}) scale(${s.scale})`}>
              <circle r={L.stoneR} fill={fill} stroke={strokeC} strokeWidth={0.9} />
              {s.color === 'B' && (
                <circle cx={-L.stoneR * 0.3} cy={-L.stoneR * 0.3} r={L.stoneR * 0.16}
                        fill="rgba(255,255,255,0.10)" />
              )}
              {s.number != null && (
                <text textAnchor="middle" dominantBaseline="central"
                      fontFamily="var(--mono)" fontSize={L.stoneR * 0.9}
                      fontWeight={600}
                      fill={s.color === 'B' ? 'var(--stone-white)' : 'var(--stone-black)'}>
                  {s.number}
                </text>
              )}
            </g>
          );
        })}

        {/* above-stone annotations */}
        {above.map((a, i) => (
          <GoAnnot key={`a-${i}`} a={a} lt={lt} L={L} n={n} />
        ))}
      </svg>
    </div>
  );
}

function GoAnnot({ a, lt, L, n }) {
  const inFade = a.fadeIn ?? 0.22;
  const outFade = a.fadeOut ?? 0.22;
  const at = a.at ?? 0;
  const until = a.until ?? Infinity;
  const opIn = clamp((lt - at) / inFade, 0, 1);
  const opOut = until === Infinity ? 1 : 1 - clamp((lt - (until - outFade)) / outFade, 0, 1);
  const op = Math.min(opIn, opOut);
  if (op <= 0.001) return null;

  const color = a.color || 'var(--accent-mcts)';

  switch (a.type) {
    case 'eye':
      return (
        <g opacity={op}>
          <circle cx={L.pos(a.x)} cy={L.pos(a.y)} r={L.stoneR * 0.7}
            fill="none" stroke={color} strokeWidth={2.2}
            strokeDasharray="4 3" />
        </g>
      );

    case 'liberty':
      return (
        <g opacity={op}>
          <circle cx={L.pos(a.x)} cy={L.pos(a.y)} r={L.stoneR * 0.24}
            fill={color} />
        </g>
      );

    case 'atari':
      return (
        <g opacity={op}>
          <circle cx={L.pos(a.x)} cy={L.pos(a.y)} r={L.stoneR * 1.2}
            fill="none" stroke={color} strokeWidth={2.2} />
        </g>
      );

    case 'ko':
      return (
        <g opacity={op}>
          <rect
            x={L.pos(a.x) - L.stoneR * 0.55}
            y={L.pos(a.y) - L.stoneR * 0.55}
            width={L.stoneR * 1.1} height={L.stoneR * 1.1}
            fill="none" stroke={color} strokeWidth={2} />
        </g>
      );

    case 'ban': {
      const r = L.stoneR * 0.55;
      const cx = L.pos(a.x), cy = L.pos(a.y);
      return (
        <g opacity={op} stroke={color} strokeWidth={2.4} strokeLinecap="round">
          <line x1={cx - r} y1={cy - r} x2={cx + r} y2={cy + r} />
          <line x1={cx + r} y1={cy - r} x2={cx - r} y2={cy + r} />
        </g>
      );
    }

    case 'territory': {
      const pts = a.points || [{ x: a.x, y: a.y }];
      const fill = a.owner === 'B' ? 'rgba(22, 19, 16, 0.32)' :
                   a.owner === 'W' ? 'rgba(247, 243, 234, 0.7)' :
                   'rgba(106, 93, 74, 0.22)';
      return (
        <g opacity={op}>
          {pts.map((p, i) => (
            <rect key={i}
              x={L.pos(p.x) - L.step * 0.46}
              y={L.pos(p.y) - L.step * 0.46}
              width={L.step * 0.92} height={L.step * 0.92}
              fill={fill} rx={2} />
          ))}
        </g>
      );
    }

    case 'dame':
      return (
        <g opacity={op}>
          <circle cx={L.pos(a.x)} cy={L.pos(a.y)} r={L.stoneR * 0.18}
            fill="var(--ink-soft)" />
        </g>
      );

    case 'label':
      return (
        <g opacity={op}>
          <text x={L.pos(a.x)} y={L.pos(a.y)}
            textAnchor="middle" dominantBaseline="central"
            fontFamily="var(--mono)" fontSize={a.size || L.stoneR * 0.8}
            fontWeight={a.weight || 500}
            fill={color}>
            {a.text}
          </text>
        </g>
      );

    case 'tip':
      return (
        <g opacity={op}>
          <text x={L.pos(a.x)} y={L.pos(a.y) - L.stoneR * 1.55}
            textAnchor="middle" fontFamily="var(--mono)" fontSize={11}
            letterSpacing="0.04em" fontWeight={500}
            fill={color}>
            {a.text}
          </text>
        </g>
      );

    case 'arrow': {
      const x1 = L.pos(a.from.x), y1 = L.pos(a.from.y);
      const x2 = L.pos(a.to.x), y2 = L.pos(a.to.y);
      const dx = x2 - x1, dy = y2 - y1, len = Math.sqrt(dx * dx + dy * dy) || 1;
      const ux = dx / len, uy = dy / len;
      const margin = L.stoneR * 0.7;
      const sx = x1 + ux * margin, sy = y1 + uy * margin;
      const ex = x2 - ux * margin, ey = y2 - uy * margin;
      const ah = 9;
      return (
        <g opacity={op}>
          <line x1={sx} y1={sy} x2={ex} y2={ey}
                stroke={color} strokeWidth={1.8} strokeLinecap="round" />
          <path d={`M ${ex} ${ey}
                    L ${ex - ux * ah - uy * ah * 0.55} ${ey - uy * ah + ux * ah * 0.55}
                    L ${ex - ux * ah + uy * ah * 0.55} ${ey - uy * ah - ux * ah * 0.55} z`}
                fill={color} />
        </g>
      );
    }

    case 'circle':
      return (
        <g opacity={op}>
          <circle cx={L.pos(a.x)} cy={L.pos(a.y)}
            r={(a.radius != null ? a.radius : 0.95) * L.stoneR}
            fill="none" stroke={color} strokeWidth={a.strokeWidth || 2}
            strokeDasharray={a.dashed ? '4 3' : undefined} />
        </g>
      );

    case 'group':
      return (
        <g opacity={op * 0.35}>
          {(a.points || []).map((p, i) => (
            <circle key={i} cx={L.pos(p.x)} cy={L.pos(p.y)} r={L.stoneR * 1.35}
              fill={color} />
          ))}
        </g>
      );

    default:
      return null;
  }
}

// ── SlideHeader ─────────────────────────────────────────────────────────────
// Standard per-slide title + optional description, with fade in.
function SlideHeader({ num, title, subtitle, maxWidth = 520 }) {
  const { localTime: lt } = useSprite();
  const op = Easing.easeOutCubic(clamp(lt / 0.5, 0, 1));
  return (
    <div style={{ opacity: op }}>
      <div style={{
        position: 'absolute', left: 200, top: 80,
      }}>
        <SectionLabel num={num} title={title} />
      </div>
      {subtitle && (
        <div style={{
          position: 'absolute', left: 200, top: 120,
          fontFamily: 'var(--serif)', fontSize: 15,
          color: 'var(--ink-soft)', maxWidth, lineHeight: 1.55,
        }}>
          {subtitle}
        </div>
      )}
    </div>
  );
}

// Animated integer counter that counts up from 0 to `value` over a window.
function CountUp({ value, at = 0, duration = 0.7, ease = Easing.easeOutCubic }) {
  const { localTime: lt } = useSprite();
  const p = ease(clamp((lt - at) / duration, 0, 1));
  return Math.round(value * p);
}

Object.assign(window, { GoAnim, GoAnnot, goLayout, SlideHeader, CountUp });
