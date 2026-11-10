import itertools

from backend import cube


def test_solved_is_valid():
    assert cube.validate_state(cube.SOLVED)
    assert cube.is_solved(cube.SOLVED)


def test_every_move_has_order_dividing_four():
    for m in cube.MOVES:
        s = cube.SOLVED
        for _ in range(4):
            s = cube.apply_move(s, m)
        assert s == cube.SOLVED, m


def test_inverse_moves_cancel():
    for m in cube.MOVES:
        s = cube.apply_moves(cube.SOLVED, [m, cube.inverse_move(m)])
        assert s == cube.SOLVED, m


def test_double_equals_two_quarters():
    for f in cube.FACES:
        assert cube.apply_moves(cube.SOLVED, [f, f]) == cube.apply_move(cube.SOLVED, f + "2")


def test_u_clockwise_moves_front_top_row_to_left():
    s = cube.apply_move(cube.SOLVED, "U")
    # F face = indices 18..26, L face = 36..44 ; top row of L now has F colour.
    assert s[36:39] == "FFF"
    assert s[18:21] == "RRR"  # right's top row comes to front


def test_rurpup_has_order_six():
    seq = ["R", "U", "R'", "U'"]
    s = cube.SOLVED
    for i in range(1, 7):
        s = cube.apply_moves(s, seq)
        if s == cube.SOLVED:
            assert i == 6
            return
    raise AssertionError("did not return to solved")


def test_states_stay_valid():
    seq, s = cube.generate_scramble(25, seed=3)
    assert cube.validate_state(s)


def test_scramble_deterministic_with_seed():
    assert cube.generate_scramble(5, seed=42) == cube.generate_scramble(5, seed=42)


def test_optimal_distance_matches_brute_force_depth2():
    for a, b in itertools.product(cube.MOVES, repeat=2):
        s = cube.apply_moves(cube.SOLVED, [a, b])
        d = cube.optimal_distance(s)
        if a[0] == b[0]:
            assert d <= 1
        else:
            assert d == 2 or (
                # commuting opposite faces still 2
                d == 2
            )


def test_optimal_distance_le_scramble_length():
    for seed in range(20):
        seq, s = cube.generate_scramble(5, seed=seed)
        d = cube.optimal_distance(s)
        assert d is not None and d <= 5


def test_optimal_distance_solved_is_zero():
    assert cube.optimal_distance(cube.SOLVED) == 0


def test_legal_moves_exclude_undo_and_revisits():
    s1 = cube.apply_move(cube.SOLVED, "R")
    allowed = cube.legal_moves(s1, {cube.SOLVED, s1})
    assert "R'" not in allowed and len(allowed) == 17


def test_correct_stickers_and_grids():
    assert cube.correct_stickers(cube.SOLVED) == 54
    assert cube.face_grids(cube.SOLVED)["U"] == ["UUU", "UUU", "UUU"]
