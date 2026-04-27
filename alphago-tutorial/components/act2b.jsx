// components/act2b.jsx
// Act 2½ — the search algorithm from Act 2, shown as pseudocode.
// Two side-by-side code blocks reflected from mcts.py.

// Left block: Node dataclass + run_mcts (the entry point).
const CODE_RUN_MCTS = `@dataclass
class Node:
    state: GameState
    N: int = 0                            # visit count
    Q: float = 0.0                        # mean action value
    logP_A: dict = {}                     # log priors from net
    children: dict = {}                   # action -> Node


def run_mcts(root_state, n_sims, config, fθ):
    """Build tree by running n_sims playouts from root_state."""
    root = Node(state=root_state)

    # Seed root priors + Dirichlet exploration noise.
    policy, _ = fθ(root_state)
    root.logP_A = {a: log(p) for a, p in policy.items()}
    add_dirichlet_noise(root, config)

    for _ in range(n_sims):
        playout(root, config, fθ)

    return root`;

// Right block: playout (the recursion) + select_action (PUCT).
const CODE_PLAYOUT = `def playout(node, config, fθ):
    """Select -> expand -> evaluate -> backup."""
    if node.state.is_terminal():
        U = node.state.reward()

    elif node.N == 0:
        # Leaf: evaluate with the neural network.
        policy, U = fθ(node.state)
        node.logP_A = {a: log(p) for a, p in policy.items()}

    else:
        # Internal: descend via PUCT.
        a = select_action(node, config.c_puct)
        if a not in node.children:
            node.children[a] = Node(state=node.state.apply(a))
        U = 1.0 - playout(node.children[a], config, fθ)

    node.N += 1
    node.Q += (U - node.Q) / node.N       # incremental mean
    return U


def select_action(node, c_puct):
    """PUCT: argmax_a  Q + c · P · √ΣN / (1 + N)"""
    sqrt_N = sqrt(sum(c.N for c in node.children.values()) + 1)

    def score(a):
        c = node.children.get(a)
        q, n = (c.Q, c.N) if c else (0, 0)
        return q + c_puct * exp(node.logP_A[a]) * sqrt_N / (1 + n)

    return max(node.logP_A, key=score)`;

// Find the first `#` character that starts an inline comment — i.e. one
// that isn't inside a quoted string.
function findCommentIdx(line) {
  let inStr = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inStr) {
      if (c === inStr) inStr = null;
    } else if (c === '"' || c === "'") {
      inStr = c;
    } else if (c === '#') {
      return i;
    }
  }
  return -1;
}

function CodeLine({ line }) {
  const trimmed = line.trim();
  // Treat a line that starts with """ or ''' as a docstring — render in muted italic.
  if (trimmed.startsWith('"""') || trimmed.startsWith("'''")) {
    return (
      <div style={{ color: 'var(--ink-soft)', fontStyle: 'italic' }}>
        {line || ' '}
      </div>
    );
  }

  const idx = findCommentIdx(line);
  if (idx < 0) return <div>{line || ' '}</div>;
  return (
    <div>
      {line.slice(0, idx)}
      <span style={{ color: 'var(--ink-soft)', fontStyle: 'italic' }}>
        {line.slice(idx)}
      </span>
    </div>
  );
}

function CodeBlock({ code, filename, style }) {
  const lines = code.split('\n');
  return (
    <div style={style}>
      {filename && (
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 10,
          letterSpacing: '0.1em', textTransform: 'uppercase',
          color: 'var(--ink-soft)', marginBottom: 6,
        }}>
          {filename}
        </div>
      )}
      <pre style={{
        fontFamily: 'var(--mono)',
        fontSize: 11,
        lineHeight: 1.4,
        color: 'var(--ink)',
        margin: 0,
        padding: '10px 14px',
        background: 'rgba(22, 19, 16, 0.04)',
        border: '1px solid rgba(31, 26, 20, 0.10)',
        borderRadius: 8,
        whiteSpace: 'pre',
        tabSize: 4,
        overflow: 'hidden',
      }}>
        {lines.map((line, i) => <CodeLine key={i} line={line} />)}
      </pre>
    </div>
  );
}

function Act2b_Pseudocode() {
  const { localTime: lt } = useSprite();
  const opL = Easing.easeOutCubic(clamp((lt - 0.2) / 0.5, 0, 1));
  const opR = Easing.easeOutCubic(clamp((lt - 0.7) / 0.5, 0, 1));

  return (
    <>
      <SlideHeader
        num="04"
        title="MCTS pseudocode"
        maxWidth={760}
      />

      <div style={{
        position: 'absolute', left: 200, top: 130, width: 510,
        opacity: opL, transform: `translateY(${(1 - opL) * 8}px)`,
      }}>
        <CodeBlock code={CODE_RUN_MCTS} filename="mcts.py · run_mcts" />
      </div>

      <div style={{
        position: 'absolute', left: 730, top: 130, width: 510,
        opacity: opR, transform: `translateY(${(1 - opR) * 8}px)`,
      }}>
        <CodeBlock code={CODE_PLAYOUT} filename="mcts.py · playout" />
      </div>
    </>
  );
}

Object.assign(window, { Act2b_Pseudocode, CodeBlock });
