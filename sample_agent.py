"""
Sample agent for Case Closed Challenge - Works with Judge Protocol
This agent runs as a Flask server and responds to judge requests.
"""

import os
from flask import Flask, request, jsonify
from collections import deque

app = Flask(__name__)

# Basic identity
PARTICIPANT = os.getenv("PARTICIPANT", "SampleParticipant")
AGENT_NAME = os.getenv("AGENT_NAME", "SampleAgent")

# Track game state
game_state = {
    "board": None,
    "agent1_trail": [],
    "agent2_trail": [],
    "agent1_length": 0,
    "agent2_length": 0,
    "agent1_alive": True,
    "agent2_alive": True,
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "turn_count": 0,
    "player_number": 1,
}


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    
    # Update our local game state
    game_state.update(data)
    
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    """
    player_number = request.args.get("player_number", default=1, type=int)
    turn_count = game_state.get("turn_count", 0)
    
    # Get our current state
    if player_number == 1:
        my_trail = game_state.get("agent1_trail", [])
        my_boosts = game_state.get("agent1_boosts", 3)
        other_trail = game_state.get("agent2_trail", [])
    else:
        my_trail = game_state.get("agent2_trail", [])
        my_boosts = game_state.get("agent2_boosts", 3)
        other_trail = game_state.get("agent1_trail", [])
    
    # Simple decision logic
    move = decide_move(my_trail, other_trail, turn_count, my_boosts)
    
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state."""
    data = request.get_json()
    if data:
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200


def decide_move(my_trail, other_trail, turn_count, my_boosts):
    """
    Heuristic Tron agent:
    - Never U-turn.
    - Score candidate moves by: (reachable space) + (Voronoi advantage) + (local freedom)
      and mild penalties for being too close early or burning boosts without benefit.
    - Uses BOOST to escape when boxed or to win a race to a big region.

    Relies on game_state["board"] (18x20 list of lists; 0 = empty, 1 = wall).  # see local-tester dummy state
    Returns "DIR" or "DIR:BOOST".  # judge expects exactly this format
    """
    # --- pull the current board (18 rows x 20 cols of 0/1) ---
    board = game_state.get("board")
    if board is None or not my_trail:
        # Fallback if no state yet
        return "RIGHT"

    H, W = len(board), (len(board[0]) if board else 0)

    # --- tiny helpers (local, no globals) ---
    def wrap(x, y):
        return x % W, y % H

    def at(x, y):
        xx, yy = wrap(x, y)
        return board[yy][xx]

    def is_empty(x, y):
        return at(x, y) == 0  # 0 == EMPTY

    DIRS = {
        "UP":    (0, -1),
        "DOWN":  (0,  1),
        "LEFT":  (-1, 0),
        "RIGHT": (1,  0),
    }
    OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

    def dir_from_trail(trail):
        if len(trail) < 2:
            return "RIGHT"
        x2, y2 = trail[-1]
        x1, y1 = trail[-2]
        dx = (x2 - x1)
        dy = (y2 - y1)
        # normalize for torus: a step can look like +/- (W-1) or (H-1)
        if dx == 1 or dx == -(W - 1):
            return "RIGHT"
        if dx == -1 or dx == (W - 1):
            return "LEFT"
        if dy == 1 or dy == -(H - 1):
            return "DOWN"
        if dy == -1 or dy == (H - 1):
            return "UP"
        return "RIGHT"

    def step_cell(x, y, dname):
        dx, dy = DIRS[dname]
        return wrap(x + dx, y + dy)

    def free_neighbors(x, y, board_like=None):
        # count/free neighbors using a board snapshot; default is current board
        M = board if board_like is None else board_like
        out = []
        for dname, (dx, dy) in DIRS.items():
            nx, ny = (x + dx) % W, (y + dy) % H
            if M[ny][nx] == 0:
                out.append((dname, nx, ny))
        return out

    def clone_board():
        return [row[:] for row in board]

    def mark_wall(M, x, y):
        xx, yy = x % W, y % H
        M[yy][xx] = 1  # 1 == AGENT/wall

    def flood_area(M, sx, sy, limit=None):
        # BFS over empty cells reachable from (sx,sy); (sx,sy) itself may be non-empty (our head),
        # so we expand into its empty neighbors.
        seen = set()
        q = []
        for _, nx, ny in free_neighbors(sx, sy, M):
            q.append((nx, ny))
            seen.add((nx, ny))
        head = 0
        count = 0
        lim = limit or (H * W)
        while head < len(q) and count < lim:
            x, y = q[head]
            head += 1
            count += 1
            for _, nx, ny in free_neighbors(x, y, M):
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return count

    def bfs_dists(M, starts):
        # multi-source BFS distances over empty cells
        from collections import deque as _dq
        INF = 10**9
        dist = [[INF] * W for _ in range(H)]
        q = _dq()
        for (sx, sy) in starts:
            dist[sy][sx] = 0
            q.append((sx, sy))
        while q:
            x, y = q.popleft()
            for _, nx, ny in free_neighbors(x, y, M):
                if dist[ny][nx] == INF:
                    dist[ny][nx] = dist[y][x] + 1
                    q.append((nx, ny))
        return dist, INF

    def torus_manhattan(ax, ay, bx, by):
        dx = min((ax - bx) % W, (bx - ax) % W)
        dy = min((ay - by) % H, (by - ay) % H)
        return dx + dy

    # --- gather live positions and direction ---
    my_head = my_trail[-1]
    opp_head = other_trail[-1] if other_trail else None
    cur_dir = dir_from_trail(my_trail)

    # candidate directions: avoid U-turn
    candidates = [d for d in ("UP", "DOWN", "LEFT", "RIGHT") if d != OPP[cur_dir]]

    # --- evaluate each (dir, maybe-boost) action ---
    best = ("RIGHT", False)
    best_score = -1e18

    for d in candidates:
        # 1-step viability
        nx1, ny1 = step_cell(*my_head, d)
        if not is_empty(nx1, ny1):
            continue  # suicide

        # simulate placing our trail for 1 step
        board1 = clone_board()
        mark_wall(board1, nx1, ny1)

        # check second step viability for boost
        boost_ok = False
        nx2 = ny2 = None
        if my_boosts > 0:
            nx2, ny2 = step_cell(nx1, ny1, d)
            if board1[ny2][nx2] == 0:
                boost_ok = True

        # define two variants to score: no-boost and (if possible) boost
        variants = [("no", (nx1, ny1), board1)]
        if boost_ok:
            board2 = [row[:] for row in board1]
            mark_wall(board2, nx2, ny2)
            variants.append(("boost", (nx2, ny2), board2))

        for mode, (hx, hy), M in variants:
            # Local freedom (branching factor) at the resulting head
            deg = len(free_neighbors(hx, hy, M))

            # Reachable area after we commit (on M)
            area = flood_area(M, hx, hy)

            # Voronoi-style advantage: cells we can reach in fewer steps than opp
            vor_adv = 0
            if opp_head is not None:
                d_me, INF = bfs_dists(M, [(hx, hy)])
                d_opp, _ = bfs_dists(M, [opp_head])
                for y in range(H):
                    for x in range(W):
                        if M[y][x] == 0:
                            dm = d_me[y][x]
                            do = d_opp[y][x]
                            if dm < do:
                                vor_adv += 1

            # Early-game “don’t get too close” penalty to avoid head-on draws
            close_pen = 0
            if opp_head is not None and turn_count < 40:
                dist = torus_manhattan(hx, hy, *opp_head)
                close_pen = max(0, 7 - dist)  # only penalize when very close

            # Prefer not to BOOST unless it clearly helps; also use BOOST to escape traps
            boost_cost = 0.8 if mode == "boost" else 0.0
            escape_bonus = 0.0
            # if we are currently in a corridor/boxed (few exits), boost gets extra credit
            cur_deg = len(free_neighbors(*my_head))
            if mode == "boost" and (cur_deg <= 1 or deg >= 3):
                escape_bonus = 3.0

            # Heuristic score (weights tuned to be sane, not perfect)
            score = (
                1.3 * area +
                1.1 * vor_adv +
                0.6 * deg -
                0.7 * close_pen -
                boost_cost +
                escape_bonus
            )

            if score > best_score:
                best_score = score
                best = (d, mode == "boost")

    # Safety fallback if somehow nothing scored
    if best is None:
        return cur_dir

    dname, use_boost = best
    return f"{dname}{':BOOST' if use_boost else ''}"



if __name__ == "__main__":
    # For development only. Port can be overridden with the PORT env var.
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
