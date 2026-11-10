"""Deterministic 3x3 Rubik's Cube engine.

State is a 54-character facelet string in the standard URFDLB order
(9 stickers per face, read row by row as seen from outside the cube).
Each character is the colour letter of the face that sticker belongs to
when solved: U R F D L B.

Move metric: half-turn metric (HTM). 18 moves; a 180 degree turn counts as 1.
Under HTM, every scramble is solvable in 20 moves or fewer ("God's number").
"""
from __future__ import annotations

import random
from collections import deque
from functools import lru_cache

FACES = "URFDLB"
SOLVED = "".join(f * 9 for f in FACES)
MOVES = [f + s for f in FACES for s in ("", "'", "2")]  # 18 moves

_NORMALS = {
    "U": (0, 1, 0),
    "R": (1, 0, 0),
    "F": (0, 0, 1),
    "D": (0, -1, 0),
    "L": (-1, 0, 0),
    "B": (0, 0, -1),
}


def _facelet_geometry():
    """Return list of (position, normal) for each of the 54 facelet indices."""
    out = []
    for face in FACES:
        for r in range(3):
            for c in range(3):
                if face == "U":
                    pos = (c - 1, 1, r - 1)
                elif face == "R":
                    pos = (1, 1 - r, 1 - c)
                elif face == "F":
                    pos = (c - 1, 1 - r, 1)
                elif face == "D":
                    pos = (c - 1, -1, 1 - r)
                elif face == "L":
                    pos = (-1, 1 - r, c - 1)
                else:  # B
                    pos = (1 - c, 1 - r, -1)
                out.append((pos, _NORMALS[face]))
    return out


_GEOM = _facelet_geometry()
_INDEX = {g: i for i, g in enumerate(_GEOM)}


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _rot_cw(v, n):
    """Rotate v by -90 degrees about n (= clockwise seen from outside of face n)."""
    c = _cross(n, v)
    d = _dot(n, v)
    return (-c[0] + n[0] * d, -c[1] + n[1] * d, -c[2] + n[2] * d)


def _build_quarter_perm(face):
    n = _NORMALS[face]
    dest = list(range(54))
    for i, (pos, nor) in enumerate(_GEOM):
        if _dot(pos, n) == 1:
            dest[i] = _INDEX[(_rot_cw(pos, n), _rot_cw(nor, n))]
    return dest


def _compose(p, q):
    """Apply p then q (new[q[p[i]]] = old[i])."""
    return [q[p[i]] for i in range(54)]


def _build_moves():
    table = {}
    for f in FACES:
        q = _build_quarter_perm(f)
        half = _compose(q, q)
        three = _compose(half, q)
        table[f] = q
        table[f + "2"] = half
        table[f + "'"] = three
    return table


_MOVE_DEST = _build_moves()
# precompute inverse form: new[j] = old[src[j]]
_MOVE_SRC = {}
for _m, _dest in _MOVE_DEST.items():
    _src = [0] * 54
    for _i, _d in enumerate(_dest):
        _src[_d] = _i
    _MOVE_SRC[_m] = tuple(_src)


def is_valid_move(move: str) -> bool:
    return move in _MOVE_SRC


def apply_move(state: str, move: str) -> str:
    src = _MOVE_SRC[move]
    return "".join(state[i] for i in src)


def apply_moves(state: str, moves) -> str:
    for m in moves:
        state = apply_move(state, m)
    return state


def inverse_move(move: str) -> str:
    if move.endswith("2"):
        return move
    if move.endswith("'"):
        return move[0]
    return move + "'"


def is_solved(state: str) -> bool:
    return state == SOLVED


def correct_stickers(state: str) -> int:
    """Number of stickers already in their solved position (54 = solved)."""
    return sum(1 for a, b in zip(state, SOLVED) if a == b)


def face_grids(state: str) -> dict:
    """3 row-strings per face, e.g. {"U": ["UUU", "UUU", "UUU"], ...}."""
    out = {}
    for fi, f in enumerate(FACES):
        base = fi * 9
        out[f] = [state[base + r * 3: base + r * 3 + 3] for r in range(3)]
    return out


def legal_moves(state: str, visited) -> list:
    """Moves that do NOT lead back to a position already seen in this attempt.

    A shortest solution never revisits a position, so removing these moves
    cannot hurt an optimal solver; it only stops R R' / U U' / cycle loops.
    Falls back to all 18 moves if (pathologically) nothing is left."""
    moves = [m for m in MOVES if apply_move(state, m) not in visited]
    return moves or list(MOVES)


def validate_state(state: str) -> bool:
    return len(state) == 54 and all(state.count(f) == 9 for f in FACES)


def generate_scramble(length: int, seed: int | None = None):
    """Random scramble with no two consecutive moves on the same face and no
    immediate A B A cancellation across opposite faces. Returns (moves, state)."""
    rng = random.Random(seed)
    opposite = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}
    seq = []
    while len(seq) < length:
        m = rng.choice(MOVES)
        if seq:
            last = seq[-1][0]
            if m[0] == last:
                continue
            if len(seq) >= 2 and seq[-2][0] == m[0] and opposite[last] == m[0]:
                continue
        seq.append(m)
    return seq, apply_moves(SOLVED, seq)


# ---------------------------------------------------------------------------
# Exact shortest-solution distance (HTM) via meet-in-the-middle BFS.
# ---------------------------------------------------------------------------

_HALF_DEPTH = 3


@lru_cache(maxsize=1)
def _solved_table():
    """BFS from the solved cube to depth 3: state -> distance."""
    table = {SOLVED: 0}
    frontier = [SOLVED]
    for d in range(1, _HALF_DEPTH + 1):
        nxt = []
        for s in frontier:
            for m in MOVES:
                t = apply_move(s, m)
                if t not in table:
                    table[t] = d
                    nxt.append(t)
        frontier = nxt
    return table


def optimal_distance(state: str):
    """Exact HTM distance to solved, if it is <= 6; otherwise None.

    Meet in the middle: BFS from `state` to depth 3, look each state up in the
    depth-3 table from the solved cube. The minimum of (depth + table depth)
    over all hits is the exact distance (valid when <= 6).
    """
    table = _solved_table()
    if state in table:
        return table[state]
    best = None
    seen = {state}
    frontier = [state]
    for d in range(1, _HALF_DEPTH + 1):
        nxt = []
        for s in frontier:
            for m in MOVES:
                t = apply_move(s, m)
                if t in seen:
                    continue
                seen.add(t)
                nxt.append(t)
                if t in table:
                    total = d + table[t]
                    if best is None or total < best:
                        best = total
        frontier = nxt
        # Any hit found at BFS depth d with table depth e gives d+e. A shorter
        # path would have been found at a smaller d or smaller e, but table
        # depth can be up to 3, so finish this level before concluding.
        if best is not None and best <= d + 1:
            break
    if best is not None and best <= 2 * _HALF_DEPTH:
        return best
    return None
