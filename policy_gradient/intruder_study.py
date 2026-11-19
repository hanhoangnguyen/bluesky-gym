"""How predictable are the competition's scripted intruders, and how many matter at once?

    python -m policy_gradient.intruder_study [n_scenarios]

Draws scenarios from the competition generator (core.scenario.ScenarioGenerator with the single-agent
defaults: 10 intruders, 5 obstacles, seed 42), flies the *neutral* policy (straight from start to goal at
150 m/s) through each one with plain kinematics, and measures, from the ego aircraft's point of view:

  * how often an intruder bends (routes have an optional mid waypoint) and by how much,
  * how wrong constant-velocity extrapolation is, as a function of the prediction horizon,
  * how often a constant-velocity conflict check misses a real conflict ("surprise conflicts"),
  * how many intruders are real threats at the same time (the case for keeping only the top K).

It needs no BlueSky: intruders follow their route polyline at constant speed and keep their last
heading after the final waypoint (an assumption about BlueSky's behaviour; not checked here). Writes
policy_gradient/figures/_intruder_study.json for the explainer page.
"""

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from core.scenario import ScenarioGenerator   # noqa: E402

C0 = (52.0, 4.0)
NM2KM = 1.852
M_PER_NM = 1852.0
HORIZONS = (60, 120, 300, 600)
DT = 10.0                      # decision interval, seconds
H_THREAT = 300                 # horizon used for "is this intruder a threat right now"
SEP = 5.0                      # NM


def to_nm(p):
    return np.array([(p[0] - C0[0]) * 60.0, (p[1] - C0[1]) * 60.0 * math.cos(math.radians(C0[0]))])


class Route:
    """Constant-speed polyline; after the last waypoint it keeps the last leg's heading."""

    def __init__(self, waypoints_ll, speed_ms):
        self.pts = np.array([to_nm(p) for p in waypoints_ll])
        self.v = speed_ms / M_PER_NM                                   # NM per second
        legs = np.diff(self.pts, axis=0)
        self.len = np.hypot(legs[:, 0], legs[:, 1])
        self.dirs = legs / self.len[:, None]
        self.cum = np.concatenate(([0.0], np.cumsum(self.len)))
        self.n_legs = len(self.len)

    def pos(self, t):
        d = np.asarray(t, dtype=float) * self.v
        leg = np.clip(np.searchsorted(self.cum, d, side="right") - 1, 0, self.n_legs - 1)
        return self.pts[leg] + self.dirs[leg] * (d - self.cum[leg])[..., None]

    def vel(self, t):
        d = np.asarray(t, dtype=float) * self.v
        leg = np.clip(np.searchsorted(self.cum, d, side="right") - 1, 0, self.n_legs - 1)
        return self.dirs[leg] * self.v

    def bend_time(self):
        return self.cum[1] / self.v if self.n_legs > 1 else None

    def bend_angle(self):
        if self.n_legs < 2:
            return None
        a = self.dirs[0] @ self.dirs[1]
        return math.degrees(math.acos(float(np.clip(a, -1, 1))))


def min_sep(ego_pos, intr_pos):
    return np.hypot(*(ego_pos - intr_pos).T).min(axis=-1)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = np.random.default_rng(42)
    gen = ScenarioGenerator(n_obstacles=5, n_intruders=10, agent_speed=150.0, center=C0)

    n_routes = n_bend = 0
    bend_angles, bend_times = [], []
    err = {h: {"all": [], "bend": []} for h in HORIZONS}
    frac_bend_within = {h: [] for h in HORIZONS}
    conf = dict(tp=0, fp=0, fn=0, tn=0)                      # CV rule vs truth, threshold 5 NM over 300 s
    n_threat_counts = []                                      # per (scenario, decision time): real threats
    n_near_counts = []                                        # per (scenario, decision time): within 30 NM
    max_threats, episode_events, episode_has_event, durations = [], [], [], []
    t0_wall = time.time()

    for sc_i in range(n):
        sc = gen.generate(rng)
        ag = sc.agents[0]
        start, goal = to_nm(ag.start), to_nm(ag.goal)
        dist = float(np.hypot(*(goal - start)))
        u = (goal - start) / dist
        v_ego = ag.speed / M_PER_NM
        t_goal = max(dist - 5.0 / NM2KM, 0.0) / v_ego         # capture radius 5 km
        durations.append(t_goal)
        ego = lambda t: start + u * (v_ego * np.minimum(np.asarray(t, dtype=float), t_goal))[..., None]
        routes = [Route(r.waypoints, r.speed) for r in sc.intruder_routes]

        for r in routes:
            n_routes += 1
            if r.n_legs > 1:
                n_bend += 1
                bend_angles.append(r.bend_angle())
                bend_times.append(r.bend_time())

        # --- constant-velocity prediction error vs horizon (observed at times spread over the flight)
        t_obs = np.arange(0.0, t_goal, 60.0)
        for r in routes:
            for h in HORIZONS:
                p_true = r.pos(t_obs + h)
                p_cv = r.pos(t_obs) + r.vel(t_obs) * h
                e = np.hypot(*(p_true - p_cv).T)
                bends = np.array([(r.n_legs > 1) and (t < r.bend_time() <= t + h) for t in t_obs])
                err[h]["all"].extend(e.tolist())
                err[h]["bend"].extend(e[bends].tolist())
                frac_bend_within[h].append(float(bends.mean()) if len(bends) else 0.0)

        # --- decision-time analysis: the ego keeps flying straight to its goal
        ts = np.arange(0.0, t_goal, DT)
        s = np.arange(DT, H_THREAT + DT, DT)                                   # look-ahead times
        events = 0
        in_conf_prev = np.zeros(len(routes), dtype=bool)
        threat_traj = np.zeros((len(ts), len(routes)), dtype=bool)
        near_traj = np.zeros((len(ts), len(routes)), dtype=bool)
        for j, r in enumerate(routes):
            T = ts[:, None] + s[None, :]                                       # (decisions, lookahead)
            e_fut = ego(T)                                                     # (D, S, 2)
            i_true = r.pos(T)
            true_min = np.hypot(e_fut[..., 0] - i_true[..., 0], e_fut[..., 1] - i_true[..., 1]).min(axis=1)
            i_cv = r.pos(ts)[:, None, :] + r.vel(ts)[:, None, :] * s[None, :, None]
            cv_min = np.hypot(e_fut[..., 0] - i_cv[..., 0], e_fut[..., 1] - i_cv[..., 1]).min(axis=1)
            threat_traj[:, j] = true_min < SEP
            cur = np.hypot(*(ego(ts) - r.pos(ts)).T)
            near_traj[:, j] = cur < 30.0
            pt, ct = true_min < SEP, cv_min < SEP
            conf["tp"] += int((pt & ct).sum()); conf["fn"] += int((pt & ~ct).sum())
            conf["fp"] += int((~pt & ct).sum()); conf["tn"] += int((~pt & ~ct).sum())
            # distinct intrusion events along the actual flight (rising edges of current separation < 5 NM)
            inside = cur < SEP
            events += int(((inside[1:] & ~inside[:-1]).sum()) + (1 if inside[0] else 0))
        n_threat_counts.extend(threat_traj.sum(axis=1).tolist())
        n_near_counts.extend(near_traj.sum(axis=1).tolist())
        max_threats.append(int(threat_traj.sum(axis=1).max()) if len(ts) else 0)
        episode_events.append(events)
        episode_has_event.append(events > 0)
        if (sc_i + 1) % 100 == 0:
            print(f"  {sc_i + 1}/{n} scenarios, {time.time() - t0_wall:.0f} s", flush=True)

    def stats(a):
        a = np.array(a)
        return dict(n=int(len(a)), mean=float(a.mean()) if len(a) else None,
                    median=float(np.median(a)) if len(a) else None,
                    p90=float(np.percentile(a, 90)) if len(a) else None)

    def hist(a, top):
        a = np.array(a)
        return [int((a == k).sum()) for k in range(top)] + [int((a >= top).sum())]

    out = dict(
        n_scenarios=n, n_routes=n_routes, share_bend=n_bend / n_routes,
        bend_angle_deg=stats(bend_angles), bend_time_s=stats(bend_times),
        bend_angle_hist=np.histogram(bend_angles, bins=[0, 10, 20, 30, 45, 60, 90, 180])[0].tolist(),
        err_by_horizon={str(h): {"all": stats(err[h]["all"]), "bend": stats(err[h]["bend"]),
                                 "share_obs_with_bend_ahead": float(np.mean(frac_bend_within[h]))} for h in HORIZONS},
        cv_rule=dict(conf, recall=conf["tp"] / max(conf["tp"] + conf["fn"], 1),
                     precision=conf["tp"] / max(conf["tp"] + conf["fp"], 1)),
        threats_hist=hist(n_threat_counts, 4), near30_hist=hist(n_near_counts, 5),
        max_threats_hist=hist(max_threats, 5),
        mean_threats=float(np.mean(n_threat_counts)), mean_near30=float(np.mean(n_near_counts)),
        neutral_episode_with_intrusion=float(np.mean(episode_has_event)),
        neutral_mean_intrusion_events=float(np.mean(episode_events)),
        mean_episode_s=float(np.mean(durations)),
    )
    path = os.path.join(HERE, "figures", "_intruder_study.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
