# train.py
# Q-table RL for Case Closed (Tron). Supports self-play and vs-heuristic.
# Saves to q_table.pkl and metadata to q_meta.json.

import argparse
import json
import os
import pickle
import random
from collections import defaultdict

from rl_utils import (
    DIRS, encode_state, all_actions, filter_actions_by_legality,
    legal_moves, next_cell, is_safe, opposite
)
from case_closed_game import Game, Direction, GameResult, AGENT, GameBoard  # AGENT used to check walls


# -----------------------------
# Merge utilities (for saving a single policy from P1/P2)
# -----------------------------
def merge_q_tables(Q1, Q2, how="avg", w1=1.0, w2=1.0):
    """
    Merge two Q dicts keyed by (state, action) -> float.
    how: 'avg' (weighted average), 'max' (elementwise max), 'p1', 'p2', 'sum'
    """
    if Q2 is None or len(Q2) == 0:
        return dict(Q1)
    if how == "p1":
        return dict(Q1)
    if how == "p2":
        return dict(Q2)

    merged = dict(Q1)  # copy
    if how == "max":
        for k, v in Q2.items():
            merged[k] = v if k not in merged else max(merged[k], v)
        return merged

    if how == "sum":
        for k, v in Q2.items():
            merged[k] = merged.get(k, 0.0) + v
        return merged

    # default: weighted average (avg)
    tot = w1 + w2 if (w1 + w2) != 0 else 1.0
    for k, v in Q2.items():
        if k in merged:
            merged[k] = (w1 * merged[k] + w2 * v) / tot
        else:
            merged[k] = v
    return merged


# -----------------------------
# Faster heuristic opponent (space + Voronoi, optimized)
# -----------------------------
def heuristic_move(game, me):
    """
    Faster heuristic:
      - No U-turns.
      - Precompute opponent distance once.
      - For each candidate (and optional BOOST), do a single constrained BFS
        that simultaneously measures reachable area and Voronoi advantage.
    Returns: (Direction, use_boost)
    """
    board = game.board
    W, H = board.width, board.height

    # Occupancy grid: 0 = empty, 1 = wall
    M = [[1 if board.get_cell_state((x, y)) == AGENT else 0
          for x in range(W)] for y in range(H)]

    my_head = me.trail[-1]
    opp = game.agent2 if me is game.agent1 else game.agent1
    opp_head = opp.trail[-1] if opp.trail else None

    DIRS = (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)
    OFFS = {
        Direction.UP:    (0, -1),
        Direction.DOWN:  (0,  1),
        Direction.LEFT:  (-1, 0),
        Direction.RIGHT: (1,  0),
    }

    def step_xy(x, y, d):
        dx, dy = OFFS[d]
        return (x + dx) % W, (y + dy) % H

    def free_deg(x, y, blocked=None, use_M=None):
        B = M if use_M is None else use_M
        bset = blocked or ()
        cnt = 0
        for d in DIRS:
            nx, ny = step_xy(x, y, d)
            if B[ny][nx] == 0 and (nx, ny) not in bset:
                cnt += 1
        return cnt

    def torus_manhattan(ax, ay, bx, by):
        dx = min((ax - bx) % W, (bx - ax) % W)
        dy = min((ay - by) % H, (by - ay) % H)
        return dx + dy

    # Opponent distance field once (on base M)
    INF = 10**9
    d_opp = [[INF] * W for _ in range(H)]
    if opp_head is not None:
        from collections import deque
        q = deque()
        ox, oy = opp_head
        if M[oy][ox] == 0:  # if opp is on an empty cell snapshot
            d_opp[oy][ox] = 0
            q.append((ox, oy))
        while q:
            x, y = q.popleft()
            nd = d_opp[y][x] + 1
            # unrolled neighbors
            nx, ny = (x + 1) % W, y
            if M[ny][nx] == 0 and d_opp[ny][nx] == INF:
                d_opp[ny][nx] = nd; q.append((nx, ny))
            nx, ny = (x - 1) % W, y
            if M[ny][nx] == 0 and d_opp[ny][nx] == INF:
                d_opp[ny][nx] = nd; q.append((nx, ny))
            nx, ny = x, (y + 1) % H
            if M[ny][nx] == 0 and d_opp[ny][nx] == INF:
                d_opp[ny][nx] = nd; q.append((nx, ny))
            nx, ny = x, (y - 1) % H
            if M[ny][nx] == 0 and d_opp[ny][nx] == INF:
                d_opp[ny][nx] = nd; q.append((nx, ny))

    # Candidate directions (no U-turn)
    cur = me.direction
    def opposite(d):
        if d == Direction.UP: return Direction.DOWN
        if d == Direction.DOWN: return Direction.UP
        if d == Direction.LEFT: return Direction.RIGHT
        return Direction.LEFT
    candidates = [d for d in DIRS if d != opposite(cur)]

    best = (cur, False)
    best_score = -1e18

    # Current branching factor (for escape bonus decision)
    cx, cy = my_head
    cur_deg = free_deg(cx, cy)

    for d in candidates:
        # Step 1 viability
        nx1, ny1 = step_xy(cx, cy, d)
        if M[ny1][nx1] != 0:
            continue

        # Check boost viability (second step)
        can_boost = False
        nx2 = ny2 = None
        if me.boosts_remaining > 0:
            nx2, ny2 = step_xy(nx1, ny1, d)
            if M[ny2][nx2] == 0 and not (nx2 == cx and ny2 == cy):
                can_boost = True

        variants = [("no", nx1, ny1)]
        if can_boost:
            variants.append(("boost", nx2, ny2))

        for mode, hx, hy in variants:
            # Cells we would newly block for this variant
            blocked = {(nx1, ny1)}
            if mode == "boost":
                blocked.add((nx2, ny2))

            # Single constrained BFS from (hx,hy):
            # - counts reachable area (ignores cells we just blocked)
            # - vor_adv++ when our dist < d_opp there
            from collections import deque
            seen = set()
            q = deque()
            area = 0
            vor_adv = 0

            # seed with neighbors of (hx,hy) that are free under this variant
            for ndx, ndy in ((hx + 1) % W, hy), ((hx - 1) % W, hy), (hx, (hy + 1) % H), (hx, (hy - 1) % H):
                if M[ndy][ndx] == 0 and (ndx, ndy) not in blocked and (ndx, ndy) not in seen:
                    seen.add((ndx, ndy))
                    q.append((ndx, ndy, 1))  # store our distance

            while q:
                x, y, dist_me = q.popleft()
                area += 1
                # Voronoi advantage test without scanning whole board
                if d_opp[y][x] > dist_me:
                    vor_adv += 1
                nd = dist_me + 1
                # Unrolled neighbors (branching kept tiny on 18x20)
                nx, ny = (x + 1) % W, y
                if M[ny][nx] == 0 and (nx, ny) not in blocked and (nx, ny) not in seen:
                    seen.add((nx, ny)); q.append((nx, ny, nd))
                nx, ny = (x - 1) % W, y
                if M[ny][nx] == 0 and (nx, ny) not in blocked and (nx, ny) not in seen:
                    seen.add((nx, ny)); q.append((nx, ny, nd))
                nx, ny = x, (y + 1) % H
                if M[ny][nx] == 0 and (nx, ny) not in blocked and (nx, ny) not in seen:
                    seen.add((nx, ny)); q.append((nx, ny, nd))
                nx, ny = x, (y - 1) % H
                if M[ny][nx] == 0 and (nx, ny) not in blocked and (nx, ny) not in seen:
                    seen.add((nx, ny)); q.append((nx, ny, nd))

            # Local branching after we commit
            deg = free_deg(hx, hy, blocked=blocked)

            # Early head-on avoidance
            close_pen = 0
            if opp_head is not None and game.turns < 40:
                close_pen = max(0, 7 - torus_manhattan(hx, hy, *opp_head))

            # Boost economics
            boost_cost = 0.6 if mode == "boost" else 0.0  # cheaper than before, but still a cost
            escape_bonus = 0.0
            if mode == "boost" and (cur_deg <= 1 or deg >= 3):
                escape_bonus = 2.5

            score = (
                1.25 * area +
                1.05 * vor_adv +
                0.55 * deg -
                0.7 * close_pen -
                boost_cost +
                escape_bonus
            )

            if score > best_score:
                best_score = score
                best = (d, mode == "boost")

    return best


# -----------------------------
# Potential-based shaping: squeeze opponent + win quickly
# -----------------------------
def _occupancy(game):
    W, H = game.board.width, game.board.height
    M = [[1 if game.board.get_cell_state((x, y)) == AGENT else 0 for x in range(W)] for y in range(H)]
    return M, W, H

def _neighbors(x, y, W, H):
    return (( (x+1) % W, y ),
            ( (x-1) % W, y ),
            ( x, (y+1) % H ),
            ( x, (y-1) % H ))

def _reachable_area(M, W, H, sx, sy):
    from collections import deque
    seen = set()
    q = deque()
    for nx, ny in _neighbors(sx, sy, W, H):
        if M[ny][nx] == 0 and (nx, ny) not in seen:
            seen.add((nx, ny))
            q.append((nx, ny))
    cnt = 0
    while q:
        x, y = q.popleft()
        cnt += 1
        for nx, ny in _neighbors(x, y, W, H):
            if M[ny][nx] == 0 and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return cnt

def space_phi(game, me):
    """Potential Φ(s) = (my_reachable - opp_reachable) / board_area."""
    opp = game.agent2 if me is game.agent1 else game.agent1
    M, W, H = _occupancy(game)
    area_me  = _reachable_area(M, W, H, *me.trail[-1])
    area_opp = _reachable_area(M, W, H, *opp.trail[-1])
    board_area = W * H
    return (area_me - area_opp) / max(1, board_area)


# -----------------------------
# Q-Learning core
# -----------------------------
class QLearner:
    def __init__(self, alpha=0.4, gamma=0.98, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, include_boost=True):
        self.Q = defaultdict(float)  # key: (state_tuple, action_tuple)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.include_boost = include_boost

    def act(self, game, me, state):
        # epsilon-greedy over legal actions
        actions = filter_actions_by_legality(game, me, all_actions(self.include_boost))
        if random.random() < self.epsilon:
            return random.choice(actions)
        # greedy
        best_a = None
        best_q = -1e9
        for a in actions:
            q = self.Q[(state, a)]
            if q > best_q:
                best_q = q
                best_a = a
        return best_a if best_a is not None else random.choice(actions)

    def update(self, s, a, r, s2, game, me_next):
        # standard Q-learning update
        old = self.Q[(s, a)]
        if s2 is None:
            target = r  # terminal
        else:
            actions_next = filter_actions_by_legality(game, me_next, all_actions(self.include_boost))
            max_next = max((self.Q[(s2, a2)] for a2 in actions_next), default=0.0)
            target = r + self.gamma * max_next
        self.Q[(s, a)] = old + self.alpha * (target - old)

    def decay_eps(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def step_reward(result_for_me, alive_step=True):
    # Base rewards:
    #   - small living cost each tick encourages faster wins / avoids loops
    #   - +1 win, -1 loss, 0 draw
    if result_for_me == "alive":
        return -0.01 if alive_step else 0.0
    if result_for_me == "win":
        return +1.0
    if result_for_me == "loss":
        return -1.0
    return 0.0  # draw / unknown


def play_one_episode(mode, learner_p1: QLearner, learner_p2: QLearner, seed=None, space_weight=0.10, quick_win_weight=0.50):
    """
    mode: 'self' or 'vs-heuristic'
    Returns: (result, steps)
    """
    if seed is not None:
        random.seed(seed)

    game = Game()
    steps = 0

    # caches for state/action per agent
    s1 = encode_state(game, game.agent1)
    s2 = encode_state(game, game.agent2)

    # Potential Φ(s) for shaping (space advantage)
    phi1_prev = space_phi(game, game.agent1)
    phi2_prev = space_phi(game, game.agent2)

    while True:
        # choose actions
        if mode == "self":
            a1 = learner_p1.act(game, game.agent1, s1)
            a2 = learner_p2.act(game, game.agent2, s2)
        elif mode == "vs-heuristic":
            a1 = learner_p1.act(game, game.agent1, s1)
            a2 = heuristic_move(game, game.agent2)
        else:
            raise ValueError("mode must be 'self' or 'vs-heuristic'")

        d1, b1 = a1
        d2, b2 = a2

        # apply step
        result = game.step(d1, d2, boost1=b1, boost2=b2)

        # compute rewards and next states
        if result is None:
            # non-terminal: both alive
            r1 = step_reward("alive", alive_step=True)
            r2 = step_reward("alive", alive_step=True)

            # potential-based shaping: r += w * (γΦ(s') - Φ(s))
            phi1_next = space_phi(game, game.agent1)
            phi2_next = space_phi(game, game.agent2)

            r1 += space_weight * (learner_p1.gamma * phi1_next - phi1_prev)
            gamma2 = learner_p2.gamma if mode == "self" else learner_p1.gamma
            r2 += space_weight * (gamma2 * phi2_next - phi2_prev)

            s1_next = encode_state(game, game.agent1)
            s2_next = encode_state(game, game.agent2)

            # updates
            learner_p1.update(s1, a1, r1, s1_next, game, game.agent1)
            if mode == "self":
                learner_p2.update(s2, a2, r2, s2_next, game, game.agent2)

            # advance
            s1, s2 = s1_next, s2_next
            phi1_prev, phi2_prev = phi1_next, phi2_next
            steps += 1
            learner_p1.decay_eps()
            if mode == "self":
                learner_p2.decay_eps()
            continue

        # terminal
        if result == GameResult.AGENT1_WIN:
            r1 = step_reward("win")
            r2 = step_reward("loss")
        elif result == GameResult.AGENT2_WIN:
            r1 = step_reward("loss")
            r2 = step_reward("win")
        else:
            # draw (including head-on)
            r1 = step_reward("draw")
            r2 = step_reward("draw")

        # quick-win bonus: earlier finishes are worth more
        W, H = game.board.width, game.board.height
        early_factor = max(0.0, 1.0 - (game.turns / max(1, W * H)))
        if result == GameResult.AGENT1_WIN:
            r1 += quick_win_weight * early_factor
        elif result == GameResult.AGENT2_WIN:
            r2 += quick_win_weight * early_factor

        # terminal shaping: Φ(s') = 0 => add -Φ(s_prev)
        r1 += space_weight * (0.0 - phi1_prev)
        r2 += space_weight * (0.0 - phi2_prev)

        # terminal updates (s' = None)
        learner_p1.update(s1, a1, r1, None, game, game.agent1)
        if mode == "self":
            learner_p2.update(s2, a2, r2, None, game, game.agent2)

        steps += 1
        return (result, steps)


def run_training(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # create learners
    common_kwargs = dict(alpha=args.alpha, gamma=args.gamma,
                         epsilon=args.eps, epsilon_min=args.eps_min,
                         epsilon_decay=args.eps_decay,
                         include_boost=not args.no_boost)

    if args.mode == "self":
        p1 = QLearner(**common_kwargs)
        p2 = QLearner(**common_kwargs)
    else:
        p1 = QLearner(**common_kwargs)
        p2 = None

    # optional warm start (LOAD, not save)
    q_path = os.path.join(args.out_dir, "q_table.pkl")
    if args.load and os.path.isfile(q_path):
        with open(q_path, "rb") as f:
            Q_loaded = pickle.load(f)
        p1.Q.update(Q_loaded)
        if p2 is not None:
            p2.Q.update(Q_loaded)
        print(f"Loaded existing Q table from {q_path} (entries: {len(Q_loaded)})")

    stats = {"episodes": 0, "p1_wins": 0, "p2_wins": 0, "draws": 0, "steps_total": 0}

    for ep in range(1, args.episodes + 1):
        result, steps = play_one_episode(
            args.mode, p1, p2 if p2 else p1,
            space_weight=args.space_weight,
            quick_win_weight=args.quick_win_weight
        )
        stats["episodes"] += 1
        stats["steps_total"] += steps
        if result == GameResult.AGENT1_WIN:
            stats["p1_wins"] += 1
        elif result == GameResult.AGENT2_WIN:
            stats["p2_wins"] += 1
        else:
            stats["draws"] += 1

        # periodic save
        if ep % args.save_every == 0 or ep == args.episodes:
            if p2 is not None:
                q_to_save = merge_q_tables(p1.Q, p2.Q, how=args.merge, w1=args.merge_w1, w2=args.merge_w2)
            else:
                q_to_save = dict(p1.Q)
            with open(q_path, "wb") as f:
                pickle.dump(q_to_save, f)
            with open(os.path.join(args.out_dir, "q_meta.json"), "w") as f:
                json.dump({
                    "episodes": stats["episodes"],
                    "p1_wins": stats["p1_wins"],
                    "p2_wins": stats["p2_wins"],
                    "draws": stats["draws"],
                    "avg_steps": stats["steps_total"]/max(1, stats["episodes"]),
                    "alpha": p1.alpha, "gamma": p1.gamma,
                    "epsilon": p1.epsilon, "epsilon_min": p1.epsilon_min,
                    "epsilon_decay": p1.epsilon_decay,
                    "include_boost": p1.include_boost,
                    "merge": args.merge, "merge_w1": args.merge_w1, "merge_w2": args.merge_w2,
                    "space_weight": args.space_weight, "quick_win_weight": args.quick_win_weight
                }, f, indent=2)
            print(f"[ep {ep}] saved Q to {q_path} | stats: {stats}")

    print("Training complete.")
    print(stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["self", "vs-heuristic"], default="vs-heuristic",
                    help="Self-play or vs a simple heuristic opponent.")
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--eps-min", type=float, default=0.05)
    ap.add_argument("--eps-decay", type=float, default=0.9995)
    ap.add_argument("--no-boost", action="store_true", help="Disable boost in action space to simplify learning.")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out-dir", type=str, default="artifacts")
    ap.add_argument("--load", action="store_true", help="Warm-start from artifacts/q_table.pkl if present.")

    # NEW: shaping knobs
    ap.add_argument("--space-weight", type=float, default=0.10,
                    help="Weight for potential-based space shaping (Φ diff).")
    ap.add_argument("--quick-win-weight", type=float, default=0.50,
                    help="Extra reward for winning earlier; scales with remaining cells.")

    # NEW: merge strategy for saving P1/P2 into one table
    ap.add_argument("--merge", choices=["avg","max","p1","p2","sum"], default="avg",
                    help="How to merge p1 and p2 Q-tables on save (self-play mode).")
    ap.add_argument("--merge-w1", type=float, default=1.0, help="Weight for p1 when --merge=avg")
    ap.add_argument("--merge-w2", type=float, default=1.0, help="Weight for p2 when --merge=avg")

    args = ap.parse_args()
    run_training(args)
