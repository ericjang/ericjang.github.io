
// animations.jsx
// Reusable animation starter: Stage, Timeline, Sprite, easing helpers.
// Usage (in an HTML file that loads React + Babel):
//
//   <Stage width={1280} height={720} duration={10} background="#f6f4ef">
//     <MyScene />
//   </Stage>
//
// Inside <Stage>, any child can call useTime() to read the current
// playhead (seconds). Or wrap content in <Sprite start={1} end={4}>...</Sprite>
// to only render during that window -- children receive a `localTime` and
// `progress` via the useSprite() hook.
//
// ─────────────────────────────────────────────────────────────────────────────

// ── Easing functions (hand-rolled, Popmotion-style) ─────────────────────────
// All easings take t ∈ [0,1] and return eased t ∈ [0,1] (may overshoot for back/elastic).
const Easing = {
  linear: (t) => t,

  // Quad
  easeInQuad:    (t) => t * t,
  easeOutQuad:   (t) => t * (2 - t),
  easeInOutQuad: (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),

  // Cubic
  easeInCubic:    (t) => t * t * t,
  easeOutCubic:   (t) => (--t) * t * t + 1,
  easeInOutCubic: (t) => (t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1),

  // Quart
  easeInQuart:    (t) => t * t * t * t,
  easeOutQuart:   (t) => 1 - (--t) * t * t * t,
  easeInOutQuart: (t) => (t < 0.5 ? 8 * t * t * t * t : 1 - 8 * (--t) * t * t * t),

  // Expo
  easeInExpo:  (t) => (t === 0 ? 0 : Math.pow(2, 10 * (t - 1))),
  easeOutExpo: (t) => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t)),
  easeInOutExpo: (t) => {
    if (t === 0) return 0;
    if (t === 1) return 1;
    if (t < 0.5) return 0.5 * Math.pow(2, 20 * t - 10);
    return 1 - 0.5 * Math.pow(2, -20 * t + 10);
  },

  // Sine
  easeInSine:    (t) => 1 - Math.cos((t * Math.PI) / 2),
  easeOutSine:   (t) => Math.sin((t * Math.PI) / 2),
  easeInOutSine: (t) => -(Math.cos(Math.PI * t) - 1) / 2,

  // Back (overshoot)
  easeOutBack: (t) => {
    const c1 = 1.70158, c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
  easeInBack: (t) => {
    const c1 = 1.70158, c3 = c1 + 1;
    return c3 * t * t * t - c1 * t * t;
  },
  easeInOutBack: (t) => {
    const c1 = 1.70158, c2 = c1 * 1.525;
    return t < 0.5
      ? (Math.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
      : (Math.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2;
  },

  // Elastic
  easeOutElastic: (t) => {
    const c4 = (2 * Math.PI) / 3;
    if (t === 0) return 0;
    if (t === 1) return 1;
    return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
  },
};

// ── Core interpolation helpers ──────────────────────────────────────────────

// Clamp a value to [min, max]
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

// interpolate([0, 0.5, 1], [0, 100, 50], ease?) -> fn(t)
// Popmotion-style: linearly maps t across input keyframes to output values,
// with optional easing per segment (single fn or array of fns).
function interpolate(input, output, ease = Easing.linear) {
  return (t) => {
    if (t <= input[0]) return output[0];
    if (t >= input[input.length - 1]) return output[output.length - 1];
    for (let i = 0; i < input.length - 1; i++) {
      if (t >= input[i] && t <= input[i + 1]) {
        const span = input[i + 1] - input[i];
        const local = span === 0 ? 0 : (t - input[i]) / span;
        const easeFn = Array.isArray(ease) ? (ease[i] || Easing.linear) : ease;
        const eased = easeFn(local);
        return output[i] + (output[i + 1] - output[i]) * eased;
      }
    }
    return output[output.length - 1];
  };
}

// animate({from, to, start, end, ease})(t) — simpler single-segment tween.
// Returns `from` before `start`, `to` after `end`.
function animate({ from = 0, to = 1, start = 0, end = 1, ease = Easing.easeInOutCubic }) {
  return (t) => {
    if (t <= start) return from;
    if (t >= end) return to;
    const local = (t - start) / (end - start);
    return from + (to - from) * ease(local);
  };
}

// ── Timeline context ────────────────────────────────────────────────────────

const TimelineContext = React.createContext({ time: 0, duration: 10, playing: false });

const useTime = () => React.useContext(TimelineContext).time;
const useTimeline = () => React.useContext(TimelineContext);

// ── Sprite ──────────────────────────────────────────────────────────────────
// Renders children only when the playhead is inside [start, end]. Provides
// a sub-context with `localTime` (seconds since start) and `progress` (0..1).
//
//   <Sprite start={2} end={5}>
//     {({ localTime, progress }) => <Thing x={progress * 100} />}
//   </Sprite>
//
// Or as a plain wrapper — children can call useSprite() themselves.

const SpriteContext = React.createContext({ localTime: 0, progress: 0, duration: 0 });
const useSprite = () => React.useContext(SpriteContext);

function Sprite({ start = 0, end = Infinity, children, keepMounted = false }) {
  const { time, duration: stageDuration } = useTimeline();
  // Half-open window [start, end). At a boundary t = end, the *next* Sprite
  // (whose start = end) takes over, so slides don't briefly double up.
  // The final Sprite (end >= stage duration) keeps its right edge inclusive
  // so it stays visible once the playhead reaches the very end.
  const atEnd = stageDuration != null && end >= stageDuration - 1e-6;
  const visible = time >= start && (atEnd ? time <= end : time < end);
  if (!visible && !keepMounted) return null;

  const duration = end - start;
  const localTime = Math.max(0, time - start);
  const progress = duration > 0 && isFinite(duration)
    ? clamp(localTime / duration, 0, 1)
    : 0;

  const value = { localTime, progress, duration, visible };

  return (
    <SpriteContext.Provider value={value}>
      {typeof children === 'function' ? children(value) : children}
    </SpriteContext.Provider>
  );
}

// ── Sample sprite components ────────────────────────────────────────────────

// TextSprite: fades/slides text in on entry, holds, then fades out on exit.
// Props: text, x, y, size, color, font, entryDur, exitDur, align
function TextSprite({
  text,
  x = 0, y = 0,
  size = 48,
  color = '#111',
  font = 'Inter, system-ui, sans-serif',
  weight = 600,
  entryDur = 0.45,
  exitDur = 0.35,
  entryEase = Easing.easeOutBack,
  exitEase = Easing.easeInCubic,
  align = 'left',
  letterSpacing = '-0.01em',
}) {
  const { localTime, duration } = useSprite();
  const exitStart = Math.max(0, duration - exitDur);

  let opacity = 1;
  let ty = 0;

  if (localTime < entryDur) {
    const t = entryEase(clamp(localTime / entryDur, 0, 1));
    opacity = t;
    ty = (1 - t) * 16;
  } else if (localTime > exitStart) {
    const t = exitEase(clamp((localTime - exitStart) / exitDur, 0, 1));
    opacity = 1 - t;
    ty = -t * 8;
  }

  const translateX = align === 'center' ? '-50%' : align === 'right' ? '-100%' : '0';

  return (
    <div style={{
      position: 'absolute',
      left: x, top: y,
      transform: `translate(${translateX}, ${ty}px)`,
      opacity,
      fontFamily: font,
      fontSize: size,
      fontWeight: weight,
      color,
      letterSpacing,
      whiteSpace: 'pre',
      lineHeight: 1.1,
      willChange: 'transform, opacity',
    }}>
      {text}
    </div>
  );
}

// ImageSprite: scales + fades in; optional Ken Burns drift during hold.
function ImageSprite({
  src,
  x = 0, y = 0,
  width = 400, height = 300,
  entryDur = 0.6,
  exitDur = 0.4,
  kenBurns = false,
  kenBurnsScale = 1.08,
  radius = 12,
  fit = 'cover',
  placeholder = null, // {label: string} for striped placeholder
}) {
  const { localTime, duration } = useSprite();
  const exitStart = Math.max(0, duration - exitDur);

  let opacity = 1;
  let scale = 1;

  if (localTime < entryDur) {
    const t = Easing.easeOutCubic(clamp(localTime / entryDur, 0, 1));
    opacity = t;
    scale = 0.96 + 0.04 * t;
  } else if (localTime > exitStart) {
    const t = Easing.easeInCubic(clamp((localTime - exitStart) / exitDur, 0, 1));
    opacity = 1 - t;
    scale = (kenBurns ? kenBurnsScale : 1) + 0.02 * t;
  } else if (kenBurns) {
    const holdSpan = exitStart - entryDur;
    const holdT = holdSpan > 0 ? (localTime - entryDur) / holdSpan : 0;
    scale = 1 + (kenBurnsScale - 1) * holdT;
  }

  const content = placeholder ? (
    <div style={{
      width: '100%', height: '100%',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'repeating-linear-gradient(135deg, #e9e6df 0 10px, #dcd8cf 10px 20px)',
      color: '#6b6458',
      fontFamily: 'JetBrains Mono, ui-monospace, monospace',
      fontSize: 13,
      letterSpacing: '0.04em',
      textTransform: 'uppercase',
    }}>
      {placeholder.label || 'image'}
    </div>
  ) : (
    <img src={src} alt="" style={{ width: '100%', height: '100%', objectFit: fit, display: 'block' }} />
  );

  return (
    <div style={{
      position: 'absolute',
      left: x, top: y,
      width, height,
      opacity,
      transform: `scale(${scale})`,
      transformOrigin: 'center',
      borderRadius: radius,
      overflow: 'hidden',
      willChange: 'transform, opacity',
    }}>
      {content}
    </div>
  );
}

// RectSprite: simple rectangle that animates position/size/color via props.
// Useful demo primitive — takes a `render` fn for per-frame customization.
function RectSprite({
  x = 0, y = 0,
  width = 100, height = 100,
  color = '#111',
  radius = 8,
  entryDur = 0.4,
  exitDur = 0.3,
  render, // optional: (ctx) => style overrides
}) {
  const spriteCtx = useSprite();
  const { localTime, duration } = spriteCtx;
  const exitStart = Math.max(0, duration - exitDur);

  let opacity = 1;
  let scale = 1;

  if (localTime < entryDur) {
    const t = Easing.easeOutBack(clamp(localTime / entryDur, 0, 1));
    opacity = clamp(localTime / entryDur, 0, 1);
    scale = 0.4 + 0.6 * t;
  } else if (localTime > exitStart) {
    const t = Easing.easeInQuad(clamp((localTime - exitStart) / exitDur, 0, 1));
    opacity = 1 - t;
    scale = 1 - 0.15 * t;
  }

  const overrides = render ? render(spriteCtx) : {};

  return (
    <div style={{
      position: 'absolute',
      left: x, top: y,
      width, height,
      background: color,
      borderRadius: radius,
      opacity,
      transform: `scale(${scale})`,
      transformOrigin: 'center',
      willChange: 'transform, opacity',
      ...overrides,
    }} />
  );
}


function Stage({
  width = 1280,
  height = 720,
  duration = 10,
  background = '#f6f4ef',
  fps = 60,
  loop = true,
  autoplay = true,
  persistKey = 'animstage',
  scrollDriven = false, // when true: wheel + arrow keys drive time, no playback bar, no RAF
  keyframes = null,     // optional array of slide-start times; enables prev-slide / play-next-slide nav
  initialTime = null,   // if provided (and finite), overrides localStorage-persisted time
  children,
}) {
  const [time, setTime] = React.useState(() => {
    if (initialTime != null && isFinite(initialTime)) {
      return clamp(initialTime, 0, duration);
    }
    try {
      const v = parseFloat(localStorage.getItem(persistKey + ':t') || '0');
      return isFinite(v) ? clamp(v, 0, duration) : 0;
    } catch { return 0; }
  });
  // Respect prefers-reduced-motion: start paused so users aren't ambushed
  // by auto-advancing motion. They can still play with the controls.
  const [playing, setPlaying] = React.useState(() => {
    if (!autoplay) return false;
    try {
      if (typeof window !== 'undefined' && window.matchMedia &&
          window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        return false;
      }
    } catch {}
    return true;
  });
  const [hoverTime, setHoverTime] = React.useState(null);
  const [scale, setScale] = React.useState(1);

  const stageRef = React.useRef(null);
  const canvasRef = React.useRef(null);
  const rafRef = React.useRef(null);
  const lastTsRef = React.useRef(null);
  // When set, the RAF loop auto-pauses once `time` reaches this value.
  // Used to implement "Down = play until next slide boundary".
  const stopAtRef = React.useRef(null);

  const kfList = React.useMemo(() => {
    if (!keyframes || keyframes.length === 0) return [0, duration];
    const set = new Set([0, ...keyframes.filter(k => k >= 0 && k <= duration), duration]);
    return Array.from(set).sort((a, b) => a - b);
  }, [keyframes, duration]);

  const prevSlideTime = React.useCallback((t) => {
    // Find the current-slide start, then return the keyframe before it.
    let curIdx = 0;
    for (let i = 0; i < kfList.length; i++) {
      if (kfList[i] <= t + 0.05) curIdx = i;
    }
    return kfList[Math.max(0, curIdx - 1)];
  }, [kfList]);

  const nextSlideTime = React.useCallback((t) => {
    // Strictly greater than t so we never return the boundary we are
    // currently parked at, but without an extra buffer that would skip the
    // very next slide when the playhead is paused at boundary − ε.
    for (const k of kfList) {
      if (k > t) return k;
    }
    return duration;
  }, [kfList, duration]);

  // The RAF loop sets stopAtRef lazily on its first frame, so we don't need
  // an eager mount-time autopause here. Doing it eagerly captured the
  // localStorage-loaded `time` and would override any pre-RAF seek (e.g.
  // from a URL-hash initializer) by yanking the playhead back.

  // Persist playhead
  React.useEffect(() => {
    try { localStorage.setItem(persistKey + ':t', String(time)); } catch {}
  }, [time, persistKey]);

  // Auto-scale to fit viewport
  React.useEffect(() => {
    if (!stageRef.current) return;
    const el = stageRef.current;
    const measure = () => {
      const barH = scrollDriven ? 0 : 44; // playback bar height
      const s = Math.min(
        el.clientWidth / width,
        (el.clientHeight - barH) / height
      );
      setScale(Math.max(0.05, s));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    window.addEventListener('resize', measure);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [width, height, scrollDriven]);

  // Animation loop. Runs whenever `playing` is true. Slides do not
  // auto-advance: while playing, the loop continually targets the current
  // slide's end boundary and pauses before the next slide becomes visible.
  // Recomputing the boundary also keeps hash/sidebar seeks from inheriting a
  // stale stop target from the previous slide.
  React.useEffect(() => {
    if (!playing) {
      lastTsRef.current = null;
      return;
    }
    const step = (ts) => {
      const firstFrame = lastTsRef.current == null;
      if (firstFrame) lastTsRef.current = ts;
      const dt = (ts - lastTsRef.current) / 1000;
      lastTsRef.current = ts;
      setTime((t) => {
        const boundary = nextSlideTime(t);
        const desiredStop = boundary != null && boundary < duration
          ? Math.max(t, boundary - 0.05)
          : null;
        const staleStop = stopAtRef.current != null && (
          stopAtRef.current < t - 1e-6 ||
          (desiredStop != null && stopAtRef.current > desiredStop + 1e-6)
        );
        if (stopAtRef.current == null || staleStop) {
          stopAtRef.current = desiredStop;
        }
        let next = t + dt;
        const stopAt = stopAtRef.current;
        if (stopAt != null && next >= stopAt) {
          next = stopAt;
          setPlaying(false);
          stopAtRef.current = null;
        } else if (next >= duration) {
          if (loop) next = next % duration;
          else { next = duration; setPlaying(false); }
        }
        return next;
      });
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      lastTsRef.current = null;
    };
  }, [playing, duration, loop, nextSlideTime]);

  // Keyboard:
  //   ↑ / PageUp   — jump to previous slide, pause
  //   ↓ / PageDown — play forward, auto-pause at next slide boundary
  //   ← / →        — step one fine keyframe, pause
  //   0 / Home     — reset, pause
  //   End          — jump to end, pause
  // Only ↓ resumes playback; everything else pauses.
  React.useEffect(() => {
    const onKey = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      const pauseNow = () => {
        setPlaying(false);
        stopAtRef.current = null;
      };
      if (e.code === 'ArrowUp' || e.code === 'PageUp') {
        e.preventDefault();
        // Jump to the start of the previous slide, then autoplay through it
        // and pause just before the current slide begins (keeps outline
        // marker on the slide that just played).
        setTime(t => {
          const prevStart = prevSlideTime(t);
          stopAtRef.current = nextSlideTime(prevStart + 0.05) - 0.05;
          return Math.max(0, prevStart + 0.01);
        });
        setPlaying(true);
      } else if (e.code === 'ArrowDown' || e.code === 'PageDown') {
        e.preventDefault();
        // Jump to the start of the next slide, then autoplay through it and
        // pause just before the slide-after begins.
        setTime(t => {
          const nextStart = nextSlideTime(t);
          stopAtRef.current = nextSlideTime(nextStart + 0.05) - 0.05;
          return Math.min(duration, nextStart + 0.01);
        });
        setPlaying(true);
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        pauseNow();
        // Tick the animation one step backward within the current slide.
        const step = e.shiftKey ? 2 : 0.5;
        setTime(t => clamp(t - step, 0, duration));
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        pauseNow();
        // Tick the animation one step forward within the current slide.
        const step = e.shiftKey ? 2 : 0.5;
        setTime(t => clamp(t + step, 0, duration));
      } else if (e.key === '0' || e.code === 'Home') {
        e.preventDefault();
        pauseNow();
        setTime(0);
      } else if (e.code === 'End') {
        e.preventDefault();
        pauseNow();
        setTime(duration);
      } else if (!scrollDriven && e.code === 'Space') {
        e.preventDefault();
        setPlaying(p => !p);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [duration, scrollDriven, prevSlideTime, nextSlideTime]);

  // Scroll-driven mode: wheel deltaY advances time, with per-frame coalescing.
  React.useEffect(() => {
    if (!scrollDriven) return;
    const el = stageRef.current;
    if (!el) return;
    let pendingDelta = 0;
    let rafId = null;
    const flush = () => {
      rafId = null;
      if (pendingDelta === 0) return;
      const d = pendingDelta;
      pendingDelta = 0;
      setTime(t => clamp(t + d, 0, duration));
    };
    const onWheel = (e) => {
      if (e.target && e.target.closest &&
          e.target.closest('button, a, input, textarea, select, summary, [data-no-timeline-wheel]')) {
        return;
      }
      e.preventDefault();
      // Manual scroll stops any ongoing autoplay.
      setPlaying(false);
      stopAtRef.current = null;
      // 0.004 s per wheel pixel → a typical ~120px tick ≈ 0.48 s (half a beat)
      pendingDelta += e.deltaY * 0.004;
      if (rafId == null) rafId = requestAnimationFrame(flush);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      el.removeEventListener('wheel', onWheel);
      if (rafId != null) cancelAnimationFrame(rafId);
    };
  }, [scrollDriven, duration]);

  const displayTime = hoverTime != null ? hoverTime : time;

  const ctxValue = React.useMemo(
    () => ({ time: displayTime, duration, playing, setTime, setPlaying }),
    [displayTime, duration, playing]
  );

  return (
    <div
      ref={stageRef}
      style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center',
        background: '#0a0a0a',
        fontFamily: 'Inter, system-ui, sans-serif',
      }}
    >
      {/* Canvas area — vertically centered in remaining space */}
      <div style={{
        flex: 1,
        width: '100%',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        overflow: 'hidden',
        minHeight: 0,
      }}>
        <div
          ref={canvasRef}
          onClick={(e) => {
            // Tap-to-navigate on touch devices: left half = prev slide,
            // right half = play next slide. Skip if the target is an
            // interactive element (button, link, sidebar).
            if (!window.matchMedia('(pointer: coarse)').matches) return;
            if (e.target.closest && e.target.closest('button, a')) return;
            const rect = canvasRef.current.getBoundingClientRect();
            const relX = e.clientX - rect.left;
            if (relX < rect.width / 2) {
              setPlaying(false);
              stopAtRef.current = null;
              setTime(t => prevSlideTime(t));
            } else {
              setTime(t => {
                stopAtRef.current = nextSlideTime(t);
                return t;
              });
              setPlaying(true);
            }
          }}
          style={{
            width, height,
            background,
            position: 'relative',
            transform: `scale(${scale})`,
            transformOrigin: 'center',
            flexShrink: 0,
            boxShadow: '0 20px 60px rgba(0,0,0,0.4)',
            overflow: 'hidden',
          }}
        >
          <TimelineContext.Provider value={ctxValue}>
            {children}
          </TimelineContext.Provider>
        </div>
      </div>

      {/* Playback bar — hidden in scroll-driven mode */}
      {!scrollDriven && (
        <PlaybackBar
          time={displayTime}
          actualTime={time}
          duration={duration}
          playing={playing}
          onPlayPause={() => setPlaying(p => !p)}
          onReset={() => { setTime(0); }}
          onSeek={(t) => setTime(t)}
          onHover={(t) => setHoverTime(t)}
        />
      )}

      {/* Scroll-driven progress rail */}
      {scrollDriven && (
        <>
          <ScrollProgress time={displayTime} duration={duration} />
          <ScrollControls
            playing={playing}
            onPlayPause={() => setPlaying(p => !p)}
            onPrev={() => {
              setPlaying(false);
              stopAtRef.current = null;
              setTime(t => Math.max(0, prevSlideTime(t)));
            }}
            onNext={() => {
              setTime(t => {
                const nextStart = nextSlideTime(t);
                stopAtRef.current = nextSlideTime(nextStart + 0.05) - 0.05;
                return Math.min(duration, nextStart + 0.01);
              });
              setPlaying(true);
            }}
          />
        </>
      )}
    </div>
  );
}

function ScrollControls({ playing, onPlayPause, onPrev, onNext }) {
  return (
    <div
      data-no-timeline-wheel
      aria-label="Timeline controls"
      style={{
        position: 'absolute',
        left: '50%',
        bottom: 14,
        transform: 'translateX(-50%)',
        zIndex: 30,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '7px 10px',
        background: 'rgba(20,20,20,0.88)',
        border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: 8,
        color: '#f6f4ef',
      }}
    >
      <IconButton onClick={onPrev} title="Previous slide" ariaLabel="Previous slide" ariaKeyshortcuts="ArrowUp">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" focusable="false">
          <path d="M9.5 2L4 7l5.5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </IconButton>
      <IconButton onClick={onPlayPause}
                  title="Play/pause"
                  ariaLabel={playing ? 'Pause' : 'Play'}
                  ariaKeyshortcuts="Space">
        {playing ? (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" focusable="false">
            <rect x="3" y="2" width="3" height="10" fill="currentColor"/>
            <rect x="8" y="2" width="3" height="10" fill="currentColor"/>
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" focusable="false">
            <path d="M3 2l9 5-9 5V2z" fill="currentColor"/>
          </svg>
        )}
      </IconButton>
      <IconButton onClick={onNext} title="Next slide" ariaLabel="Next slide" ariaKeyshortcuts="ArrowDown">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" focusable="false">
          <path d="M4.5 2L10 7l-5.5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </IconButton>
    </div>
  );
}

// ── Scroll progress rail ────────────────────────────────────────────────────
// Thin progress bar pinned to the top of the viewport in scroll-driven mode.
function ScrollProgress({ time, duration }) {
  const pct = duration > 0 ? clamp(time / duration, 0, 1) * 100 : 0;
  return (
    <div style={{
      position: 'absolute', top: 0, left: 0, right: 0,
      height: 3, pointerEvents: 'none', zIndex: 20,
      background: 'rgba(31, 26, 20, 0.08)',
    }}>
      <div style={{
        width: `${pct}%`, height: '100%',
        background: 'var(--ink, #1f1a14)',
        transition: 'width 80ms linear',
      }} />
    </div>
  );
}

// ── Playback bar ────────────────────────────────────────────────────────────
// Play/pause, return-to-begin, scrub track, time display.
// Uses fixed-width time fields so layout doesn't thrash.

function PlaybackBar({ time, duration, playing, onPlayPause, onReset, onSeek, onHover }) {
  const trackRef = React.useRef(null);
  const [dragging, setDragging] = React.useState(false);

  const timeFromEvent = React.useCallback((e) => {
    const rect = trackRef.current.getBoundingClientRect();
    const x = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    return x * duration;
  }, [duration]);

  const onTrackMove = (e) => {
    if (!trackRef.current) return;
    const t = timeFromEvent(e);
    if (dragging) {
      onSeek(t);
    } else {
      onHover(t);
    }
  };

  const onTrackLeave = () => {
    if (!dragging) onHover(null);
  };

  const onTrackDown = (e) => {
    setDragging(true);
    const t = timeFromEvent(e);
    onSeek(t);
    onHover(null);
  };

  React.useEffect(() => {
    if (!dragging) return;
    const onUp = () => setDragging(false);
    const onMove = (e) => {
      if (!trackRef.current) return;
      const t = timeFromEvent(e);
      onSeek(t);
    };
    window.addEventListener('mouseup', onUp);
    window.addEventListener('mousemove', onMove);
    return () => {
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('mousemove', onMove);
    };
  }, [dragging, timeFromEvent, onSeek]);

  const pct = duration > 0 ? (time / duration) * 100 : 0;
  const fmt = (t) => {
    const total = Math.max(0, t);
    const m = Math.floor(total / 60);
    const s = Math.floor(total % 60);
    const cs = Math.floor((total * 100) % 100);
    return `${String(m).padStart(1, '0')}:${String(s).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
  };

  const mono = 'JetBrains Mono, ui-monospace, SFMono-Regular, monospace';

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '8px 16px',
      background: 'rgba(20,20,20,0.92)',
      borderTop: '1px solid rgba(255,255,255,0.08)',
      width: '100%',
      maxWidth: 680,
      alignSelf: 'center',

      borderRadius: 8,
      color: '#f6f4ef',
      fontFamily: 'Inter, system-ui, sans-serif',
      userSelect: 'none',
      flexShrink: 0,
    }}>
      <IconButton onClick={onReset}
                  title="Return to start (0)"
                  ariaLabel="Return to start"
                  ariaKeyshortcuts="0">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" focusable="false">
          <path d="M3 2v10M12 2L5 7l7 5V2z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round"/>
        </svg>
      </IconButton>
      <IconButton onClick={onPlayPause}
                  title="Play/pause (space)"
                  ariaLabel={playing ? 'Pause' : 'Play'}
                  ariaKeyshortcuts="Space">
        {playing ? (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" focusable="false">
            <rect x="3" y="2" width="3" height="10" fill="currentColor"/>
            <rect x="8" y="2" width="3" height="10" fill="currentColor"/>
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" focusable="false">
            <path d="M3 2l9 5-9 5V2z" fill="currentColor"/>
          </svg>
        )}
      </IconButton>

      {/* Current time: fixed width so it doesn't thrash */}
      <div style={{
        fontFamily: mono,
        fontSize: 12,
        fontVariantNumeric: 'tabular-nums',
        width: 64, textAlign: 'right',
        color: '#f6f4ef',
      }}>
        {fmt(time)}
      </div>

      {/* Scrub track. Real <input type=range> would be ideal, but we paint a
          custom track behind the playhead — so we expose ARIA semantics on
          the wrapper and handle key events directly. */}
      <div
        ref={trackRef}
        role="slider"
        tabIndex={0}
        aria-label="Timeline"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration * 100) / 100}
        aria-valuenow={Math.round(time * 100) / 100}
        aria-valuetext={`${fmt(time)} of ${fmt(duration)}`}
        onMouseMove={onTrackMove}
        onMouseLeave={onTrackLeave}
        onMouseDown={onTrackDown}
        onKeyDown={(e) => {
          const big = e.shiftKey ? 5 : 1;
          if (e.key === 'ArrowLeft')      { e.preventDefault(); onSeek(Math.max(0, time - big)); }
          else if (e.key === 'ArrowRight') { e.preventDefault(); onSeek(Math.min(duration, time + big)); }
          else if (e.key === 'Home')       { e.preventDefault(); onSeek(0); }
          else if (e.key === 'End')        { e.preventDefault(); onSeek(duration); }
          else if (e.key === 'PageDown')   { e.preventDefault(); onSeek(Math.min(duration, time + 10)); }
          else if (e.key === 'PageUp')     { e.preventDefault(); onSeek(Math.max(0, time - 10)); }
        }}
        style={{
          flex: 1,
          height: 22,
          position: 'relative',
          cursor: 'pointer',
          display: 'flex', alignItems: 'center',
        }}
      >
        <div style={{
          position: 'absolute',
          left: 0, right: 0, height: 4,
          background: 'rgba(255,255,255,0.12)',
          borderRadius: 2,
        }}/>
        <div style={{
          position: 'absolute',
          left: 0, width: `${pct}%`, height: 4,
          background: 'oklch(72% 0.12 250)',
          borderRadius: 2,
        }}/>
        <div style={{
          position: 'absolute',
          left: `${pct}%`, top: '50%',
          width: 12, height: 12,
          marginLeft: -6, marginTop: -6,
          background: '#fff',
          borderRadius: 6,
          boxShadow: '0 2px 4px rgba(0,0,0,0.4)',
        }}/>
      </div>

      {/* Duration: fixed width */}
      <div style={{
        fontFamily: mono,
        fontSize: 12,
        fontVariantNumeric: 'tabular-nums',
        width: 64, textAlign: 'left',
        color: 'rgba(246,244,239,0.55)',
      }}>
        {fmt(duration)}
      </div>
    </div>
  );
}

function IconButton({ children, onClick, title, ariaLabel, ariaKeyshortcuts }) {
  const [hover, setHover] = React.useState(false);
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-label={ariaLabel || title}
      aria-keyshortcuts={ariaKeyshortcuts}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        width: 28, height: 28,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: hover ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.04)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: 6,
        color: '#f6f4ef',
        cursor: 'pointer',
        padding: 0,
        transition: 'background 120ms',
      }}
    >
      {children}
    </button>
  );
}


Object.assign(window, {
  Easing, interpolate, animate, clamp,
  TimelineContext, useTime, useTimeline,
  Sprite, SpriteContext, useSprite,
  TextSprite, ImageSprite, RectSprite,
  Stage, PlaybackBar,
});
