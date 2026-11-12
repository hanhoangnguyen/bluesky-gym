"""One-step actor-critic with a Gaussian policy (Sutton & Barto, Ch. 13, Sec. 13.5 and 13.7),
tile-coded features, on top of NavEnv.

Ported from the Dubins-car version in ~/policy_optimization/actor_critic.py. The algorithm and the
update rules are unchanged; what is new is that the features come from a `State` (goal, nearest
obstacle, nearest intruder) instead of the car's pose, and the evaluation reports the competition
metrics (goal reached, restricted-area entries, intrusions, flight time).

    pi(a | s) = Normal(mu(s), sigma(s)^2)    mu = theta_mu . x(s)    sigma = exp(theta_sigma . x(s))
    v(s)      = w . x(s)

x(s) is the concatenation of several tile-coded groups (each group has n_tilings active tiles):
    goal      (distance to goal, goal bearing)                                      always
    obstacle  (clearance, bearing)  and  (goal bearing, bearing, clearance)         stage >= 2
    intruder  (distance, bearing, aspect)                                           stage >= 3
All angles are relative to the aircraft's own heading, so what is learned is the same everywhere in
the sector.
"""

import math
import random

import numpy as np

from policy_gradient.scenarios import StageGenerator, canonical_scenario


class TileCoder:
    """Tile coding (book Section 9.5.4). `n_tilings` shifted grids lie over the input space; a point is
    in exactly one tile of each, so x(s) is all zeros except n_tilings ones and we store only the indices.

    lows, highs, bins: range and number of tiles per input dimension.
    wrap:              True for an angle (the last tile connects back to the first).
    """

    def __init__(self, lows, highs, bins, wrap, n_tilings):
        dims = len(lows)
        self.lows = np.array(lows, dtype=float)
        self.widths = (np.array(highs, dtype=float) - self.lows) / np.array(bins)
        self.wrap = np.array(wrap, dtype=bool)
        self.tiles = np.array(bins) + np.where(self.wrap, 0, 1)    # shifted grids stick out by one tile
        self.offsets = (np.arange(n_tilings)[:, None] * (2 * np.arange(dims)[None, :] + 1) / n_tilings) % 1.0
        self.strides = np.array([int(np.prod(self.tiles[j + 1:])) for j in range(dims)])
        per_tiling = int(np.prod(self.tiles))
        self.bases = np.arange(n_tilings) * per_tiling
        self.size = n_tilings * per_tiling

    def indices(self, point):
        cells = np.floor((np.asarray(point) - self.lows) / self.widths + self.offsets).astype(int)
        cells = np.where(self.wrap, cells % self.tiles, np.clip(cells, 0, self.tiles - 1))
        return cells @ self.strides + self.bases


class Features:
    """Maps a NavEnv `State` to the indices of the active tiles."""

    FAR_NM = 60.0     # "no obstacle / intruder nearby" is encoded as a far-away one

    def __init__(self, stage, n_tilings=8, cpa=False):
        pi = math.pi
        self.coders = [(self._goal, TileCoder((0, -pi), (160, pi), (16, 12), (False, True), n_tilings))]
        if stage >= 2:
            self.coders.append((self._obstacle, TileCoder((-5, -pi), (50, pi), (11, 12), (False, True), n_tilings)))
            self.coders.append((self._obstacle_joint, TileCoder((-pi, -pi, -5), (pi, pi, 40), (8, 8, 5),
                                                                (True, True, False), n_tilings)))
        if stage >= 3:
            self.coders.append((self._intruder, TileCoder((0, -pi, -pi), (80, pi, pi), (8, 8, 8),
                                                          (False, True, True), n_tilings)))
            if cpa:   # appended last, so a checkpoint trained without it still fills the front
                self.coders.append((self._cpa, TileCoder((0, -15), (400, 15), (10, 15), (False, False), n_tilings)))
                self.coders.append((self._cpa_joint, TileCoder((0, -15, -pi), (400, 15, pi), (8, 10, 8),
                                                               (False, False, True), n_tilings)))
        sizes = [c.size for _, c in self.coders]
        self.offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        self.size = int(sum(sizes))
        self.n_active = len(self.coders) * n_tilings

    @staticmethod
    def _goal(s):
        return (s.dist_goal_km, s.goal_bearing)

    def _obstacle(self, s):
        clearance, bearing = s.obstacle if s.obstacle else (self.FAR_NM, 0.0)
        return (clearance, bearing)

    def _obstacle_joint(self, s):
        clearance, bearing = s.obstacle if s.obstacle else (self.FAR_NM, 0.0)
        return (s.goal_bearing, bearing, clearance)

    def _intruder(self, s):
        return s.intruder[:3] if s.intruder else (self.FAR_NM + 20, 0.0, math.pi)

    @staticmethod
    def _cpa(s):
        """(time to closest approach, signed miss distance): the quantities a conflict is decided by."""
        return s.intruder[3:5] if s.intruder else (400.0, 15.0)

    def _cpa_joint(self, s):
        tcpa, dcpa = self._cpa(s)
        return (tcpa, dcpa, s.goal_bearing)

    def indices(self, state):
        return np.concatenate([coder.indices(point(state)) + off
                               for (point, coder), off in zip(self.coders, self.offsets)])


class ActorCritic:
    def __init__(self, env, stage, gamma=1.0, alpha_w=0.3, alpha_mu=0.3, alpha_sigma=0.02,
                 sigma_init=0.5, sigma_min=0.05, n_tilings=8, seed=None, n_eval=20, cpa=False):
        self.env, self.stage = env, stage
        self.gamma, self.alpha_w, self.alpha_mu, self.alpha_sigma = gamma, alpha_w, alpha_mu, alpha_sigma
        self.log_sigma_min, self.log_sigma_max = math.log(sigma_min), math.log(1.0)
        self.rng = random.Random(seed)

        self.features = Features(stage, n_tilings, cpa)
        n = self.features.size
        self.w = np.zeros(n)                                                    # v(s) = 0
        self.theta_mu = np.zeros(n)                                             # mu(s) = 0: keep heading
        self.theta_sigma = np.full(n, math.log(sigma_init) / self.features.n_active)

        self.history, self.snapshots = [], {}
        gen = StageGenerator(stage, fixed_prob=0.0, scale=env.scale)
        eval_rng = np.random.default_rng(12345)
        self.eval_scenarios = [gen.generate(eval_rng) for _ in range(n_eval)]   # fixed, comparable evaluations

    # ----- policy ------------------------------------------------------------------------
    def mu_sigma(self, idx):
        mu = self.theta_mu[idx].sum()
        log_sigma = min(max(self.theta_sigma[idx].sum(), self.log_sigma_min), self.log_sigma_max)
        return mu, math.exp(log_sigma)

    def act(self, state, greedy=False):
        mu, sigma = self.mu_sigma(self.features.indices(state))
        return mu if greedy else self.rng.gauss(mu, sigma)

    def rollout(self, scenario, greedy=True, record=None):
        """One episode with the current policy. Returns the final `info` dict (the scored metrics).
        `record`, if given, is a list that gets one (state, action) pair per step."""
        state = self.env.reset(scenario)
        terminal = truncated = False
        info = self.env.info
        while not (terminal or truncated):
            a = self.act(state, greedy)
            if record is not None:
                record.append((state, a))
            state, _, terminal, truncated, info = self.env.step(a)
        return info

    def evaluate(self):
        """Greedy drives on the canonical layout and on the fixed set of random scenarios."""
        rec = []
        canon = self.rollout("canonical", record=rec)
        infos = [self.rollout(sc) for sc in self.eval_scenarios]
        reached = [i for i in infos if i["waypoint_reached"]]
        n = len(infos)
        # sigma along the canonical drive (how much the policy still explores)
        sigma = sum(self.mu_sigma(self.features.indices(s))[1] for s, _ in rec) / max(len(rec), 1)
        return dict(
            canon_reached=canon["waypoint_reached"], canon_time=canon["flight_time"],
            canon_restricted=canon["restricted_area_events"], canon_intrusions=canon["intrusion_events"],
            success=len(reached) / n,
            restricted=sum(i["restricted_area_events"] > 0 for i in infos) / n,
            intrusion=sum(i["intrusion_events"] > 0 for i in infos) / n,
            sector_exit=sum(i["sector_exit_events"] > 0 for i in infos) / n,
            mean_time=sum(i["flight_time"] for i in reached) / len(reached) if reached else float("nan"),
            sigma=sigma)

    # ----- parameters ----------------------------------------------------------------------
    def get_params(self):
        return dict(w=self.w.copy(), theta_mu=self.theta_mu.copy(), theta_sigma=self.theta_sigma.copy())

    def set_params(self, p):
        self.w[:], self.theta_mu[:], self.theta_sigma[:] = p["w"], p["theta_mu"], p["theta_sigma"]

    @staticmethod
    def _push(theta, idx, step, total, low, high):
        """theta[idx] += step, unless `total` is already outside [low, high] and the step goes further out
        (keeps the mean within ~2 sigma of the action limit and sigma within its bounds)."""
        if (total >= high and step > 0) or (total <= low and step < 0):
            return
        theta[idx] += step

    # ----- training ------------------------------------------------------------------------
    def train(self, num_episodes, eval_every=500, snapshot_at=(), verbose=True):
        """One-step actor-critic (episodic), book p. 332. Scenarios come from the env's random
        StageGenerator (which already mixes in the canonical layout with its `fixed_prob`)."""
        env, gamma, w = self.env, self.gamma, self.w
        recent = []
        if 0 in snapshot_at:
            self.snapshots[0] = self.get_params()

        for episode in range(1, num_episodes + 1):                    # loop for each episode
            state = env.reset()                                       # S, I <- 1
            idx, I, ret = self.features.indices(state), 1.0, 0.0
            terminal = truncated = False

            while not (terminal or truncated):                        # loop for each step
                mu, sigma = self.mu_sigma(idx)
                a = self.rng.gauss(mu, sigma)                         # A ~ pi(.|S, theta)
                state, R, terminal, truncated, _ = env.step(a)        # take A, observe S', R
                idx_next = self.features.indices(state)

                v = w[idx].sum()
                v_next = 0.0 if terminal else w[idx_next].sum()       # v(S') = 0 at the goal
                delta = R + gamma * v_next - v                        # TD error

                n = len(idx)
                z = (a - mu) / sigma
                w[idx] += self.alpha_w / n * delta
                # grad_mu ln pi = (a-mu)/sigma^2 x(s); scaled by sigma^2 (natural gradient for the mean)
                # so the step is (a-mu) x(s) -- see the Dubins version for why plain gradient collapsed.
                limit = 1.0 + 2 * sigma
                self._push(self.theta_mu, idx, self.alpha_mu / n * I * delta * z * sigma, mu, -limit, limit)
                self._push(self.theta_sigma, idx, self.alpha_sigma / n * I * delta * (z * z - 1),
                           self.theta_sigma[idx].sum(), self.log_sigma_min, self.log_sigma_max)

                I *= gamma
                idx = idx_next
                ret += R

            recent.append(ret)
            if episode in snapshot_at:
                self.snapshots[episode] = self.get_params()
            if episode % eval_every == 0:
                result = self.evaluate()
                result.update(episode=episode, train_return=sum(recent) / len(recent))
                self.history.append(result)
                recent = []
                if verbose:
                    print(f"ep {episode:6d} | train ret {result['train_return']:6.2f} | eval: "
                          f"goal {result['success']:4.0%}, restricted {result['restricted']:4.0%}, "
                          f"intrusion {result['intrusion']:4.0%}, exit {result['sector_exit']:4.0%}, "
                          f"time {result['mean_time']:5.0f}s | canonical: "
                          f"{'goal' if result['canon_reached'] else 'MISS'} {result['canon_time']:.0f}s "
                          f"restr {result['canon_restricted']} intr {result['canon_intrusions']} | "
                          f"sigma {result['sigma']:.2f}", flush=True)
