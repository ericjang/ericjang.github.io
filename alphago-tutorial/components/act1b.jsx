// components/act1b.jsx
// Act 1½ — raw policy vs. search. Phase A: pick a single top-policy
// successor. Phase B: branch into three vertically-stacked successors,
// then expand the centre one through another policy head into three
// grandchildren — each leaf has a value-head readout.

function Act1b_WhySearch() {
  const { localTime: lt } = useSprite();
  const op = (at, dur = 0.6) =>
    Easing.easeOutCubic(clamp((lt - at) / dur, 0, 1));

  // Phase timings (slide window ~7 s)
  const rootOp         = op(0.2);
  const policyOp       = op(0.8);
  const arrowInOp      = op(1.1);
  const arrowOutSingle = op(1.5);
  const singleSuccOp   = op(1.7);
  const branchT        = clamp((lt - 2.7) / 0.9, 0, 1);   // 0 = single, 1 = 3-stack
  const valueOp        = op(3.7);
  const policy2Op      = op(4.3);
  const grandOp        = op(4.8);
  const grandValueOp   = op(5.4);
  const policyLabelOp  = op(5.6);
  const valueLabelOp   = op(5.6);

  // Board contents — a simple 3×3 position with 3 plausible white replies.
  const rootStones = [
    { x: 0, y: 0, color: 'B' },
    { x: 2, y: 0, color: 'W' },
    { x: 0, y: 2, color: 'B' },
  ];
  const reply = (x, y) => [...rootStones, { x, y, color: 'W' }];
  const SUCC = [
    { stones: reply(2, 1), hl: { x: 2, y: 1 }, v: 0.62 },
    { stones: reply(1, 2), hl: { x: 1, y: 2 }, v: 0.48 },
    { stones: reply(2, 2), hl: { x: 2, y: 2 }, v: 0.27 },
  ];
  const TOP = SUCC[0];

  // Grandchildren expanded from the center successor — each gets a value.
  const midStones = SUCC[1].stones;
  const grandReply = (x, y) => [...midStones, { x, y, color: 'B' }];
  const GRAND = [
    { stones: grandReply(1, 1), hl: { x: 1, y: 1 }, v: 0.55 },
    { stones: grandReply(2, 2), hl: { x: 2, y: 2 }, v: 0.41 },
    { stones: grandReply(0, 1), hl: { x: 0, y: 1 }, v: 0.33 },
  ];

  // ── Layout ───────────────────────────────────────────────────────────────
  const rootX = 220, rootY = 270, rootSize = 140;
  const policyX = 410, policyY = 295, policyW = 100, policyH = 90;
  const policyCenterY = policyY + policyH / 2;
  const policyOutX = policyX + policyW;
  const policyOutY = policyCenterY;

  // Phase A destination (single successor, fades out as branches appear)
  const singleX = 580, singleY = 270, singleSize = 140;

  // Phase B: 3 successors stacked vertically
  const succX = 580;
  const succSize = 96;
  const succYs = [150, 295, 440];     // top / middle / bottom
  const succXR = succX + succSize;     // right edge

  // Second policy head — coming out of the middle successor
  const policy2X = 770, policy2Y = 305, policy2W = 84, policy2H = 76;
  const policy2CY = policy2Y + policy2H / 2;

  // Grandchildren
  const grandX = 920;
  const grandSize = 78;
  const grandYs = [180, 310, 440];

  return (
    <>
      <SlideHeader num="02" title="Raw policy vs. search" maxWidth={860} subtitle={<>
        You <em>could</em> just play whatever the policy network recommends directly. Lookahead search, guided by the same network, boosts the same model into a much stronger player.
      </>} />

      {/* Root board */}
      <div style={{
        position: 'absolute', left: rootX, top: rootY,
        opacity: rootOp,
        transform: `translateY(${(1 - rootOp) * 6}px)`,
      }}>
        <GoBoard n={3} size={rootSize} stones={rootStones} />
        <Caption mono size={11} style={{ marginTop: 8, textAlign: 'center', width: rootSize }}>
          current state
        </Caption>
      </div>

      {/* Policy head glyph (1) */}
      <div style={{
        position: 'absolute', left: policyX, top: policyY,
        opacity: policyOp,
        transform: `scale(${0.9 + 0.1 * policyOp})`,
        transformOrigin: 'center',
      }}>
        <div style={{
          width: policyW, height: policyH,
          background: 'var(--bg)',
          border: '1px dashed var(--accent-net)',
          borderRadius: 10,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <NetGlyph width={72} height={52} color="var(--accent-net)" active />
        </div>
        <Caption mono size={10} color="var(--accent-net)"
                 style={{ marginTop: 6, textAlign: 'center', width: policyW }}>
          f<sub>θ</sub>
        </Caption>
      </div>

      {/* Phase A: single successor */}
      <div style={{
        position: 'absolute', left: singleX, top: singleY,
        opacity: singleSuccOp * (1 - branchT),
        transform: `translateY(${(1 - singleSuccOp) * 6}px)`,
      }}>
        <GoBoard n={3} size={singleSize} stones={TOP.stones}
                 highlights={[{ x: TOP.hl.x, y: TOP.hl.y, color: 'var(--accent-net)' }]} />
        <Caption mono size={11} style={{ marginTop: 8, textAlign: 'center', width: singleSize }}>
          top policy move
        </Caption>
      </div>

      {/* Phase B: three vertically-stacked successors */}
      {branchT > 0 && SUCC.map((b, i) => {
        const supersededByChildren = i === 1 ? grandOp : 0;
        return (
        <div key={i} style={{
          position: 'absolute',
          left: succX, top: succYs[i],
          opacity: branchT,
          transform: `translateX(${(1 - branchT) * -10}px)`,
        }}>
          <GoBoard n={3} size={succSize} stones={b.stones}
                   highlights={[{ x: b.hl.x, y: b.hl.y, color: 'var(--accent-net)' }]} />
          <div style={{
            position: 'absolute', left: succSize + 8, top: succSize / 2 - 9,
            opacity: valueOp * (1 - 0.55 * supersededByChildren),
            fontFamily: 'var(--mono)', fontSize: 13,
            color: supersededByChildren > 0
              ? `color-mix(in oklch, var(--accent-mcts) ${100 - 100 * supersededByChildren}%, var(--ink-soft))`
              : 'var(--accent-mcts)',
            fontWeight: 600,
            textDecoration: supersededByChildren > 0.5 ? 'line-through' : 'none',
          }}>
            v = {b.v.toFixed(2)}
          </div>
        </div>
        );
      })}

      {/* Policy head (2) — coming out of the centre successor */}
      <div style={{
        position: 'absolute', left: policy2X, top: policy2Y,
        opacity: policy2Op,
        transform: `scale(${0.9 + 0.1 * policy2Op})`,
        transformOrigin: 'center',
      }}>
        <div style={{
          width: policy2W, height: policy2H,
          background: 'var(--bg)',
          border: '1px dashed var(--accent-net)',
          borderRadius: 9,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <NetGlyph width={58} height={42} color="var(--accent-net)" active />
        </div>
        <Caption mono size={9.5} color="var(--accent-net)"
                 style={{ marginTop: 4, textAlign: 'center', width: policy2W }}>
          f<sub>θ</sub>
        </Caption>
      </div>

      {/* Grandchildren — three boards stacked vertically with values */}
      {GRAND.map((g, i) => (
        <div key={i} style={{
          position: 'absolute',
          left: grandX, top: grandYs[i],
          opacity: grandOp,
          transform: `translateX(${(1 - grandOp) * -10}px)`,
        }}>
          <GoBoard n={3} size={grandSize} stones={g.stones}
                   highlights={[{ x: g.hl.x, y: g.hl.y, color: 'var(--accent-mcts)' }]} />
          <div style={{
            position: 'absolute', left: grandSize + 6, top: grandSize / 2 - 8,
            opacity: grandValueOp,
            fontFamily: 'var(--mono)', fontSize: 12,
            color: 'var(--accent-mcts)', fontWeight: 600,
          }}>
            v = {g.v.toFixed(2)}
          </div>
        </div>
      ))}

      {/* SVG overlay: arrows + connector lines for labels */}
      <svg style={{
        position: 'absolute', inset: 0,
        pointerEvents: 'none',
      }} width="100%" height="100%">
        {/* Root → policy 1 */}
        <ArrowCurve
          x1={rootX + rootSize} y1={rootY + rootSize / 2}
          x2={policyX - 4}      y2={policyCenterY}
          arrow strokeWidth={1.3}
          color="var(--ink)"
          progress={arrowInOp}
          curvature={0}
        />

        {/* Phase A: policy 1 → single successor */}
        <g opacity={1 - branchT}>
          <ArrowCurve
            x1={policyOutX + 4} y1={policyOutY}
            x2={singleX - 4}    y2={singleY + singleSize / 2}
            arrow strokeWidth={1.3}
            color="var(--accent-net)"
            progress={arrowOutSingle}
            curvature={0}
          />
        </g>

        {/* Phase B: policy 1 → 3 vertically-stacked successors */}
        <g opacity={branchT}>
          {SUCC.map((_, i) => {
            const targetY = succYs[i] + succSize / 2;
            const dy = targetY - policyOutY;
            const cur = clamp(dy / 320, -0.45, 0.45);
            return (
              <ArrowCurve
                key={i}
                x1={policyOutX + 4} y1={policyOutY}
                x2={succX - 2}      y2={targetY}
                arrow strokeWidth={1.2}
                color="var(--accent-net)"
                progress={1}
                curvature={cur}
              />
            );
          })}
        </g>

        {/* Centre successor → policy 2 */}
        <g opacity={policy2Op}>
          <ArrowCurve
            x1={succXR + 6}      y1={succYs[1] + succSize / 2}
            x2={policy2X - 4}    y2={policy2CY}
            arrow strokeWidth={1.2}
            color="var(--accent-net)"
            progress={1}
            curvature={0}
          />
        </g>

        {/* Policy 2 → 3 grandchildren */}
        <g opacity={grandOp}>
          {GRAND.map((_, i) => {
            const targetY = grandYs[i] + grandSize / 2;
            const dy = targetY - policy2CY;
            const cur = clamp(dy / 260, -0.45, 0.45);
            return (
              <ArrowCurve
                key={i}
                x1={policy2X + policy2W + 4} y1={policy2CY}
                x2={grandX - 2}              y2={targetY}
                arrow strokeWidth={1.1}
                color="var(--accent-net)"
                progress={1}
                curvature={cur}
              />
            );
          })}
        </g>

        {/* Connector — "policy network" label → policy 1 glyph */}
        <g opacity={policyLabelOp}>
          <path
            d={`M 350 600 Q 420 560 ${policyX + policyW / 2} ${policyY + policyH + 14}`}
            stroke="var(--accent-net)" strokeWidth={1} fill="none"
            strokeDasharray="3 3" strokeLinecap="round" />
        </g>

        {/* Connectors — "value network" label → every value readout
            (both the three successors AND the three grandchildren). */}
        <g opacity={valueLabelOp}>
          {[...SUCC.map((_, i) => ({
              tx: succX + succSize + 56, ty: succYs[i] + succSize / 2,
            })),
            ...GRAND.map((_, i) => ({
              tx: grandX + grandSize + 50, ty: grandYs[i] + grandSize / 2,
            })),
          ].map((t, i) => {
            const labelX = 1010, labelY = 600;
            const ctrlX = (labelX + t.tx) / 2;
            const ctrlY = labelY - 30;
            return (
              <path key={`vl${i}`}
                d={`M ${labelX} ${labelY} Q ${ctrlX} ${ctrlY} ${t.tx} ${t.ty}`}
                stroke="var(--accent-mcts)" strokeWidth={1} fill="none"
                strokeDasharray="3 3" strokeLinecap="round" />
            );
          })}
        </g>
      </svg>

      {/* "policy network" label */}
      <div style={{
        position: 'absolute', left: 200, top: 575,
        width: 280,
        opacity: policyLabelOp,
      }}>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 11,
          letterSpacing: '0.1em', textTransform: 'uppercase',
          color: 'var(--accent-net)', marginBottom: 4,
        }}>
          policy network
        </div>
        <div style={{ fontFamily: 'var(--serif)', fontSize: 13.5, color: 'var(--ink)', lineHeight: 1.5 }}>
          Picks which boards are worth reaching — pruning most of the branching factor away.
        </div>
      </div>

      {/* "value network" label */}
      <div style={{
        position: 'absolute', right: 60, top: 575,
        width: 320, textAlign: 'right',
        opacity: valueLabelOp,
      }}>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 11,
          letterSpacing: '0.1em', textTransform: 'uppercase',
          color: 'var(--accent-mcts)', marginBottom: 4,
        }}>
          value network
        </div>
        <div style={{ fontFamily: 'var(--serif)', fontSize: 13.5, color: 'var(--ink)', lineHeight: 1.5 }}>
          Scores each reachable board without having to play the game to the end.
        </div>
      </div>

    </>
  );
}

Object.assign(window, { Act1b_WhySearch });
