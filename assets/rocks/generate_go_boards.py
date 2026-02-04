#!/usr/bin/env python3
"""Validate 3x3 Go board positions for the alphago-mcts scrollytelling scene.

Models simplified 3x3 Go rules (placement, capture via flood-fill, suicide check).
Defines a tree of board positions, validates each is legal, and checks that each
child differs from its parent by exactly one stone.

Prints validated board states + policy/value data as JSON to stdout.
"""

import json
import sys

# Board is a flat list of 9 cells (3x3), indexed row-major:
#   0 1 2
#   3 4 5
#   6 7 8
# Values: 0=empty, 1=black, 2=white

NEIGHBORS = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4, 6],
    4: [1, 3, 5, 7],
    5: [2, 4, 8],
    6: [3, 7],
    7: [4, 6, 8],
    8: [5, 7],
}


def get_group(board, pos):
    """Flood-fill to find all stones in the group containing `pos`."""
    color = board[pos]
    if color == 0:
        return set()
    visited = set()
    stack = [pos]
    while stack:
        p = stack.pop()
        if p in visited:
            continue
        if board[p] != color:
            continue
        visited.add(p)
        stack.extend(NEIGHBORS[p])
    return visited


def liberties(board, group):
    """Count liberties (adjacent empty cells) for a group."""
    libs = set()
    for p in group:
        for n in NEIGHBORS[p]:
            if board[n] == 0:
                libs.add(n)
    return libs


def is_legal(board):
    """Check that no group on the board has zero liberties."""
    checked = set()
    for i in range(9):
        if board[i] == 0 or i in checked:
            continue
        group = get_group(board, i)
        checked |= group
        if len(liberties(board, group)) == 0:
            return False
    return True


def diff_count(parent, child):
    """Count cells that differ between parent and child boards."""
    return sum(1 for a, b in zip(parent, child) if a != b)


def diff_positions(parent, child):
    """Return list of (index, parent_val, child_val) for differing cells."""
    return [(i, a, b) for i, (a, b) in enumerate(zip(parent, child))
            if a != b]


# ─── Tree definition ────────────────────────────────────────

ROOT = [2, 0, 1, 0, 1, 0, 2, 0, 0]  # Black to play

A  = [2, 1, 1, 0, 1, 0, 2, 0, 0]    # B plays cell 1
B  = [2, 0, 1, 0, 1, 0, 2, 0, 1]    # B plays cell 8
C  = [2, 0, 1, 1, 1, 0, 2, 0, 0]    # B plays cell 3

B1 = [2, 2, 1, 0, 1, 0, 2, 0, 1]    # W plays cell 1
B2 = [2, 0, 1, 2, 1, 0, 2, 0, 1]    # W plays cell 3

TREE = {
    "root": {
        "board": ROOT,
        "children": ["A", "B", "C"],
        "turn": "black",
        "policy": [0.0, 0.30, 0.0, 0.25, 0.0, 0.10, 0.0, 0.05, 0.30],
    },
    "A": {
        "board": A,
        "parent": "root",
        "children": [],
        "turn": "white",
        "value": 0.72,
        "policy": [0.0, 0.0, 0.0, 0.35, 0.0, 0.25, 0.0, 0.20, 0.20],
    },
    "B": {
        "board": B,
        "parent": "root",
        "children": ["B1", "B2"],
        "turn": "white",
        "policy": [0.0, 0.35, 0.0, 0.30, 0.0, 0.15, 0.0, 0.10, 0.0],
    },
    "C": {
        "board": C,
        "parent": "root",
        "children": [],
        "turn": "white",
        "value": 0.65,
        "policy": [0.0, 0.30, 0.0, 0.0, 0.0, 0.30, 0.0, 0.20, 0.20],
    },
    "B1": {
        "board": B1,
        "parent": "B",
        "children": [],
        "turn": "black",
        "value": 0.55,
        "policy": [0.0, 0.0, 0.0, 0.40, 0.0, 0.30, 0.0, 0.15, 0.0],
    },
    "B2": {
        "board": B2,
        "parent": "B",
        "children": [],
        "turn": "black",
        "value": 0.38,
        "policy": [0.0, 0.25, 0.0, 0.0, 0.0, 0.35, 0.0, 0.25, 0.0],
    },
}


def validate():
    errors = []

    for name, node in TREE.items():
        board = node["board"]

        # Check legality
        if not is_legal(board):
            errors.append(f"{name}: illegal position (group with zero liberties)")

        # Check parent diff
        parent_name = node.get("parent")
        if parent_name:
            parent_board = TREE[parent_name]["board"]
            diffs = diff_positions(parent_board, board)
            if len(diffs) != 1:
                errors.append(
                    f"{name}: differs from parent {parent_name} by "
                    f"{len(diffs)} cells (expected 1): {diffs}"
                )
            elif diffs[0][1] != 0:
                errors.append(
                    f"{name}: changed cell {diffs[0][0]} was not empty in "
                    f"parent (was {diffs[0][1]})"
                )

    return errors


def main():
    errors = validate()
    if errors:
        print("VALIDATION ERRORS:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    print("All positions validated successfully.", file=sys.stderr)

    # Output JSON
    output = {}
    for name, node in TREE.items():
        entry = {
            "board": node["board"],
            "turn": node["turn"],
            "policy": node["policy"],
            "children": node["children"],
        }
        if "value" in node:
            entry["value"] = node["value"]
        if "parent" in node:
            entry["parent"] = node["parent"]
            parent_board = TREE[node["parent"]]["board"]
            # Find the move (changed cell index)
            diffs = diff_positions(parent_board, node["board"])
            entry["move"] = diffs[0][0]
        output[name] = entry

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
