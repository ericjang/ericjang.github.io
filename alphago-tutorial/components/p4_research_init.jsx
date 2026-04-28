// components/p4_research_init.jsx
// Part 4 · slide 12 — research initialization framing.
// Re-renders the AlphaGo Zero training-time vs Elo curve from a JSON of
// sampled data points so we can place the circle annotation at the data
// coordinate (40 hrs, 3900 Elo) instead of guessing pixel offsets on top
// of the source bitmap.

const RESEARCH_INIT_DATA_URL = '/alphago-tutorial/alphago_zero_progression.json';

function P4_ResearchInit() {
  const { localTime: lt } = useSprite();
  const headerOp = Easing.easeOutCubic(clamp(lt / 0.5, 0, 1));
  const figureOp = Easing.easeOutCubic(clamp((lt - 0.4) / 0.6, 0, 1));
  const circleOp = Easing.easeOutCubic(clamp((lt - 1.4) / 0.6, 0, 1));
  const noteOp   = Easing.easeOutCubic(clamp((lt - 1.9) / 0.6, 0, 1));

  // Stage opacities (one per annotation `stage` value in the JSON).
  const stageOps = {
    1: Easing.easeOutCubic(clamp((lt - 1.4) / 0.6, 0, 1)),
    2: Easing.easeOutCubic(clamp((lt - 2.4) / 0.6, 0, 1)),
  };

  const [data, setData] = React.useState(null);
  React.useEffect(() => {
    let cancelled = false;
    fetch(RESEARCH_INIT_DATA_URL)
      .then(r => r.json())
      .then(j => { if (!cancelled) setData(j); })
      .catch((err) => { console.error('failed to load plot data', err); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', flexDirection: 'column',
      paddingLeft: 200, paddingRight: 60,
      paddingTop: 76, paddingBottom: 56,
      gap: 18,
    }}>
      <div style={{ opacity: headerOp }}>
        <SectionLabel num="15" title="Always start from a good init" />
        <div style={{
          marginTop: 12,
          fontFamily: 'var(--serif)', fontSize: 15,
          color: 'var(--ink-soft)',
          maxWidth: 920, lineHeight: 1.55,
        }}>
          In any research project I like to "initialize as close to the final outcome" as we can, to cut down on iteration time and make sure that we don't make mistakes even when everything is handed to us on a silver platter. 
          Pure-RL self-play (AlphaGo Zero, blue) crosses the AlphaGo Lee baseline only after roughly 30 hours of training, so we want to skip this slow part and focus on the fast iteration loops to reach success. After we get the easy win, it is much easier to backtrack to a harder initialization and get it to recover the original performance.
        </div>
      </div>

      <div style={{
        flex: '1 1 auto', minHeight: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {data && (
          <ProgressionPlot
            data={data}
            figureOp={figureOp}
            stageOps={stageOps}
          />
        )}
      </div>
    </div>
  );
}

// SVG plot. All data → pixel mapping happens here: pass in a JSON shaped
// like `alphago_zero_progression.json` and the annotations land on the
// data-space coordinates regardless of stage scale.
function ProgressionPlot({ data, figureOp, stageOps }) {
  // viewBox dimensions and plot-area margins.
  const W = 760, H = 560;
  const M = { top: 18, right: 24, bottom: 50, left: 78 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const xMin = data.axes.x.min, xMax = data.axes.x.max;
  const yMin = data.axes.y.min, yMax = data.axes.y.max;

  const xScale = (x) => M.left + ((x - xMin) / (xMax - xMin)) * plotW;
  const yScale = (y) => M.top + (1 - (y - yMin) / (yMax - yMin)) * plotH;

  const xTicks = [0, 10, 20, 30, 40, 50, 60, 70];
  const yTicks = [-4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000];

  const lineSeries  = data.series.filter(s => s.kind === 'line');
  const hlineSeries = data.series.filter(s => s.kind === 'hline');
  const seriesById  = Object.fromEntries(data.series.map(s => [s.id, s]));

  const polyline = (pts) =>
    pts.map(([x, y]) => `${xScale(x).toFixed(2)},${yScale(y).toFixed(2)}`).join(' ');

  // Returns the subset of `points` covering [fromX, toX], with the endpoints
  // linearly interpolated so the highlighted segment lines up exactly with
  // the source curve. Lets the JSON specify range bounds in data coords
  // without snapping to sampled points.
  const sliceSeries = (points, fromX, toX) => {
    const interp = (x) => {
      for (let i = 1; i < points.length; i++) {
        const [x0, y0] = points[i - 1];
        const [x1, y1] = points[i];
        if (x >= Math.min(x0, x1) && x <= Math.max(x0, x1)) {
          const t = (x - x0) / (x1 - x0 || 1);
          return [x, y0 + t * (y1 - y0)];
        }
      }
      return null;
    };
    const inside = points.filter(([x]) => x > fromX && x < toX);
    const start = interp(fromX);
    const end = interp(toX);
    const out = [];
    if (start) out.push(start);
    out.push(...inside);
    if (end) out.push(end);
    return out;
  };

  const annotations = data.annotations || [];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      style={{
        width: '100%', height: '100%',
        opacity: figureOp, transition: 'opacity 240ms ease',
        fontFamily: 'var(--mono)',
      }}
    >
      {/* Plot frame */}
      <rect
        x={M.left} y={M.top} width={plotW} height={plotH}
        fill="rgba(31,26,20,0.02)"
        stroke="rgba(31,26,20,0.18)" strokeWidth={1}
      />

      {/* Y gridlines + labels */}
      {yTicks.map(t => {
        const y = yScale(t);
        return (
          <g key={`y${t}`}>
            <line
              x1={M.left} x2={M.left + plotW} y1={y} y2={y}
              stroke="rgba(31,26,20,0.08)" strokeWidth={0.6}
            />
            <text
              x={M.left - 8} y={y + 3.5}
              textAnchor="end" fontSize={11}
              fill="var(--ink-soft)" fontVariantNumeric="tabular-nums"
            >
              {t.toLocaleString()}
            </text>
          </g>
        );
      })}

      {/* X tick marks + labels */}
      {xTicks.map(t => {
        const x = xScale(t);
        return (
          <g key={`x${t}`}>
            <line
              x1={x} x2={x} y1={M.top + plotH} y2={M.top + plotH + 4}
              stroke="rgba(31,26,20,0.4)" strokeWidth={0.8}
            />
            <text
              x={x} y={M.top + plotH + 18}
              textAnchor="middle" fontSize={11}
              fill="var(--ink-soft)" fontVariantNumeric="tabular-nums"
            >
              {t}
            </text>
          </g>
        );
      })}

      {/* Axis labels */}
      <text
        x={M.left + plotW / 2} y={H - 12}
        textAnchor="middle" fontFamily="var(--serif)" fontSize={13}
        fill="var(--ink)"
      >
        {data.axes.x.label}
      </text>
      <text
        x={18} y={M.top + plotH / 2}
        textAnchor="middle" fontFamily="var(--serif)" fontSize={13}
        fill="var(--ink)"
        transform={`rotate(-90 18 ${M.top + plotH / 2})`}
      >
        {data.axes.y.label}
      </text>

      {/* Horizontal baseline series (e.g., AlphaGo Lee) */}
      {hlineSeries.map(s => {
        const y = yScale(s.y);
        return (
          <line
            key={s.id}
            x1={M.left} x2={M.left + plotW} y1={y} y2={y}
            stroke={s.color} strokeWidth={2}
            strokeDasharray={s.dashed ? '6 5' : undefined}
          />
        );
      })}

      {/* Curve series */}
      {lineSeries.map(s => (
        <polyline
          key={s.id}
          points={polyline(s.points)}
          fill="none" stroke={s.color} strokeWidth={2.2}
          strokeLinejoin="round" strokeLinecap="round"
        />
      ))}

      {/* Legend, lower right inside plot area */}
      {(() => {
        const lx = M.left + plotW - 188;
        const ly = M.top + plotH - 12 - data.series.length * 18;
        return (
          <g>
            <rect
              x={lx - 10} y={ly - 14}
              width={188} height={data.series.length * 18 + 14}
              fill="rgba(247,243,234,0.92)"
              stroke="rgba(31,26,20,0.10)" strokeWidth={0.8}
              rx={3}
            />
            {data.series.map((s, i) => {
              const ry = ly + i * 18;
              return (
                <g key={s.id}>
                  <line
                    x1={lx} x2={lx + 24} y1={ry} y2={ry}
                    stroke={s.color} strokeWidth={2.2}
                    strokeDasharray={s.kind === 'hline' && s.dashed ? '5 4' : undefined}
                  />
                  <text
                    x={lx + 30} y={ry + 4}
                    fontSize={11} fontFamily="var(--serif)"
                    fill="var(--ink)"
                  >
                    {s.label}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })()}

      {/* Highlight segments — render before markers so the marker stroke
          sits cleanly on top of the highlighted curve. The dasharray /
          dashoffset trick (with pathLength=1) draws the line on from the
          marker outward as the stage progress ramps from 0 → 1. */}
      {annotations.map(a => {
        const series = seriesById[a.seriesId];
        if (!series || series.kind !== 'line' || !a.range) return null;
        const seg = sliceSeries(series.points, a.range.fromX, a.range.toX);
        const stageOp = stageOps[a.stage] != null ? stageOps[a.stage] : 1;
        const color = a.color || 'var(--accent-mcts)';
        return (
          <polyline
            key={`hl-${a.id}`}
            points={polyline(seg)}
            fill="none" stroke={color} strokeWidth={4.6}
            strokeLinejoin="round" strokeLinecap="round"
            pathLength={1}
            strokeDasharray="1 1"
            strokeDashoffset={1 - stageOp}
            opacity={stageOp > 0 ? 1 : 0}
          />
        );
      })}

      {/* Markers + callout cards, one per annotation. */}
      {annotations.map(a => {
        const stageOp = stageOps[a.stage] != null ? stageOps[a.stage] : 1;
        const color = a.color || 'var(--accent-mcts)';
        // Marker + callout fade in alongside the line draw, slightly
        // delayed so the line lands first.
        const reveal = clamp((stageOp - 0.4) / 0.6, 0, 1);
        const mx = xScale(a.marker.x);
        const my = yScale(a.marker.y);
        const noteX = a.note?.x != null ? xScale(a.note.x) : mx + 24;
        const noteY = a.note?.y != null ? yScale(a.note.y) : my - 16;
        const noteW = a.note?.width || 240;
        return (
          <g key={`ann-${a.id}`} opacity={reveal}>
            <circle
              cx={mx} cy={my} r={9}
              fill="var(--bg)" stroke={color} strokeWidth={3}
            />
            <foreignObject
              x={noteX} y={noteY}
              width={noteW} height={120}
            >
              <div xmlns="http://www.w3.org/1999/xhtml" style={{
                padding: '8px 10px',
                fontFamily: 'var(--serif)', fontSize: 13,
                color: 'var(--ink)', lineHeight: 1.4,
                background: 'rgba(246,241,231,0.95)',
                border: `1px solid ${color}`,
                borderLeftWidth: 3,
                borderRadius: 4,
                boxShadow: '0 6px 14px rgba(31,26,20,0.10)',
              }}>
                <div style={{
                  fontFamily: 'var(--mono)', fontSize: 9.5,
                  letterSpacing: '0.14em', textTransform: 'uppercase',
                  color, fontWeight: 600,
                  marginBottom: 4,
                }}>
                  {a.label}
                </div>
                {a.note?.text}
              </div>
            </foreignObject>
          </g>
        );
      })}
    </svg>
  );
}

Object.assign(window, { P4_ResearchInit });
