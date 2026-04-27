"""Monte Carlo Tree Search — AlphaGo-style pseudocode.

A distilled version of the production implementation, kept next to the
animated tutorial so the slide and the code stay in sync. The full
implementation (with batching, rollout mixing, transposition tables, and
the trace machinery used for debugging) lives in autogo/mcts/.
"""

from dataclasses import dataclass, field
from math import log, exp, sqrt


@dataclass
class Node:
    state: GameState
    N: int = 0                                      # visit count
    Q: float = 0.0                                  # root (mean action) value
    logP_A: dict = field(default_factory=dict)      # log priors from net
    children: dict = field(default_factory=dict)    # action -> Node


def run_mcts(root_state, n_sims, config, fθ):
    """Build tree by running n_sims playouts from root_state."""
    root = Node(state=root_state)

    # Seed root priors + Dirichlet exploration noise.
    policy, _ = fθ(root_state)
    root.logP_A = {a: log(p) for a, p in policy.items()}
    add_dirichlet_noise(root, config)

    for _ in range(n_sims):
        playout(root, config, fθ)

    return root


def playout(node, config, fθ):
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
    node.Q += (U - node.Q) / node.N            # incremental mean
    return U


def select_action(node, c_puct):
    """PUCT: argmax_a  Q + c · P · √ΣN / (1 + N)"""
    sqrt_N = sqrt(sum(c.N for c in node.children.values()) + 1)

    def score(a):
        c = node.children.get(a)
        q, n = (c.Q, c.N) if c else (0, 0)
        return q + c_puct * exp(node.logP_A[a]) * sqrt_N / (1 + n)

    return max(node.logP_A, key=score)
