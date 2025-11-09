
from case_closed_game import Direction, GameBoard, AGENT 

DIRS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
DIR_TO_IDX = {Direction.UP: 0, Direction.DOWN: 1, Direction.LEFT: 2, Direction.RIGHT: 3}

def opposite(d: Direction) -> Direction:
    if d == Direction.UP: return Direction.DOWN
    if d == Direction.DOWN: return Direction.UP
    if d == Direction.LEFT: return Direction.RIGHT
    return Direction.LEFT

def torus_delta(a: int, b: int, size: int) -> int:
    raw = b - a
    half = size // 2
    if raw > half:
        raw -= size
    elif raw < -half:
        raw += size
    return raw

def next_cell(board: GameBoard, pos: tuple[int, int], direction: Direction) -> tuple[int, int]:
    dx, dy = direction.value
    return board._torus_check((pos[0] + dx, pos[1] + dy))

def is_safe(board: GameBoard, cell: tuple[int, int]) -> bool:
    return board.get_cell_state(cell) != AGENT

def legal_moves(game, me, forbid_reverse: bool = True):
    head = me.trail[-1]
    cur = me.direction
    moves = []
    for d in DIRS:
        if forbid_reverse:
            cur_dx, cur_dy = cur.value
            req_dx, req_dy = d.value
            if (req_dx, req_dy) == (-cur_dx, -cur_dy):
                continue
        if is_safe(game.board, next_cell(game.board, head, d)):
            moves.append(d)
    if not moves:
        moves = [cur]
    return moves

def all_actions(allow_boost: bool = True):
    if allow_boost:
        return [(d, False) for d in DIRS] + [(d, True) for d in DIRS]
    return [(d, False) for d in DIRS]

def filter_actions_by_legality(game, me, actions):
    legal_dirs = set(legal_moves(game, me, forbid_reverse=True))
    out = []
    for d, b in actions:
        if d in legal_dirs and (not b or me.boosts_remaining > 0):
            out.append((d, b))
    if not out:
        out = [(me.direction, False)]
    return out

def _free_neighbors_deg(board: GameBoard, x: int, y: int) -> int:
    deg = 0
    for d in DIRS:
        nx, ny = next_cell(board, (x, y), d)
        if is_safe(board, (nx, ny)):
            deg += 1
    return deg

def _frontier_open_len(board: GameBoard, start: tuple[int, int], d: Direction, max_len: int = 4) -> int:
    x, y = start
    length = 0
    for _ in range(max_len):
        x, y = next_cell(board, (x, y), d)
        if not is_safe(board, (x, y)):
            break
        length += 1
    return length

def _torus_manhattan(board: GameBoard, a: tuple[int,int], b: tuple[int,int]) -> int:
    dx = abs(torus_delta(a[0], b[0], board.width))
    dy = abs(torus_delta(a[1], b[1], board.height))
    return dx + dy


def encode_state(game, me):
    board = game.board
    opp = game.agent2 if me is game.agent1 else game.agent1

    my_head = me.trail[-1]
    opp_head = opp.trail[-1]

    # direction index + boosts
    my_dir_idx = DIR_TO_IDX[me.direction]
    my_boosts = max(0, min(3, me.boosts_remaining))

    safety_bits = 0
    neigh = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    for i, d in enumerate(neigh):
        nx, ny = next_cell(board, my_head, d)
        safety_bits |= (1 if is_safe(board, (nx, ny)) else 0) << i

    dx = torus_delta(my_head[0], opp_head[0], board.width)
    dy = torus_delta(my_head[1], opp_head[1], board.height)
    dx_enc = max(-5, min(5, dx)) + 5
    dy_enc = max(-5, min(5, dy)) + 5

    corridor_deg = _free_neighbors_deg(board, my_head[0], my_head[1])  # 0..4

    prox = _torus_manhattan(board, my_head, opp_head)
    opp_prox_bucket = min(6, prox)


    frontier_open = _frontier_open_len(board, my_head, me.direction, max_len=4)

    return (my_dir_idx, my_boosts, safety_bits, dx_enc, dy_enc,
            corridor_deg, opp_prox_bucket, frontier_open)
