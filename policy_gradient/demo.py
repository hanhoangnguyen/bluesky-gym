"""Demo for the 1:1: fly the same scenario with a baseline and with the learned policy on the real
CompetitionEnv, then replay both side by side with the live quantities the agent is looking at.

    python -m policy_gradient.demo                        # slide + animated dashboard (opens windows)
    python -m policy_gradient.demo --scenario 12          # pick another of the 100 evaluation scenarios
    python -m policy_gradient.demo --save demo.gif        # write the animation instead of showing it
    python -m policy_gradient.demo --find                 # scan the 100 scenarios, list good demo candidates
    python -m policy_gradient.demo --baseline policy_gradient/checkpoints/stage3_cpa.npz
                                                           # compare against the heading-only policy

Both runs are recorded in the gym first (BlueSky allows one environment per process), then replayed.
What you see per column: the map (sector, obstacle and the enclosing circle the shipped observation
describes, intruder with its 5 NM bubble), the straight-line closest-approach prediction that our
features are built from, and a readout of the numbers the policy sees and the action it sends.
These are our toy scenarios (square obstacle, one straight-line intruder, half scale), not the
competition scenarios.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

import bluesky as bs
import bluesky_gym.envs.common.functions as fn

from policy_gradient.nav_env import NavEnv
from policy_gradient.actor_critic import ActorCritic

CK = "policy_gradient/checkpoints/"
LEARNED = CK + "stage3_speed.npz"
HERE = os.path.dirname(os.path.abspath(__file__))
GATE_T, GATE_D = 300.0, 8.0          # "threat" = will pass within 8 NM within 300 s


# ------------------------------------------------------------------------------ policies
def load_policy(env, path, gate=None):
    ck = np.load(path)
    speed = "theta_mu_s" in ck.files
    for cpa in (True, False):                      # old checkpoints were trained without the CPA features
        ac = ActorCritic(env, 3, seed=0, n_eval=100, cpa=cpa, speed=speed, gate=gate)
        if ac.w.shape[0] >= ck["w"].shape[0]:
            break
    n = ck["w"].shape[0]
    ac.w[:n], ac.theta_mu[:n], ac.theta_sigma[:n] = ck["w"], ck["theta_mu"], ck["theta_sigma"]
    if speed:
        ac.theta_mu_s[:n], ac.theta_sigma_s[:n] = ck["theta_mu_s"], ck["theta_sigma_s"]
    return ac


def policy_fn(ac):
    def f(state):
        a = ac.act(state, greedy=True)
        return a, dict(v=float(ac.w[ac.features.indices(state)].sum()))
    return f


def pursuit_fn(speed):
    """Hand-coded baseline, not learned: turn toward the goal, ignore obstacles and traffic."""
    def f(state):
        a = max(-1.0, min(1.0, state.goal_bearing / math.radians(45.0)))
        return ((a, 0.0) if speed else a), {}
    return f


# ------------------------------------------------------------------------------ recording
def geometry(env):
    inner = env.env
    sc, c = inner.scenario, np.array(inner.scenario.center)

    def ne(ll):
        n, e = fn.latlong_to_nm(c, np.array(ll))
        return (float(e), float(n))               # plot x = east, y = north

    obst = []
    for ob in sc.obstacles:
        poly = np.array([ne(v) for v in ob.vertices])
        cen = poly.mean(axis=0)
        obst.append(dict(poly=poly.tolist(), centre=cen.tolist(),
                         radius=float(np.max(np.hypot(*(poly - cen).T)))))
    return dict(sector=[ne(v) for v in sc.sector], obstacles=obst, goal=ne(sc.agents[0].goal),
                start=ne(sc.agents[0].start), n_intruders=len(sc.intruder_routes))


def run(env, act_fn, scenario, speed):
    """One episode, greedy. Returns (geometry, frames, final info); a frame per decision."""
    env.speed = speed
    state = env.reset(scenario)
    inner, info = env.env, env.info
    centre = np.array(inner.scenario.center)
    geo = geometry(env)
    last_ego = [None]

    def ne(i):
        n, e = fn.latlong_to_nm(centre, np.array([bs.traf.lat[i], bs.traf.lon[i]]))
        return (float(e), float(n))

    def snap(state, a, aux, info):
        i = bs.traf.id2idx(inner.agent)
        if i >= 0:
            last_ego[0] = (ne(i), float(bs.traf.hdg[i]), float(bs.traf.tas[i]))
        ego, hdg, tas = last_ego[0]
        intr = []
        for iid in inner.intruder_ids:
            j = bs.traf.id2idx(iid)
            intr.append((*ne(j), float(bs.traf.hdg[j]), float(bs.traf.tas[j])) if j >= 0 else None)
        return dict(t=float(info["flight_time"]), ego=ego, hdg=hdg, tas=tas, intr=intr, state=state, a=a, aux=aux,
                    m={k: float(info[k]) for k in ("intrusion_time", "time_in_restricted_area", "intrusion_events",
                                                   "restricted_area_events", "sector_exit_events", "waypoint_reached")})

    frames, terminal, truncated = [], False, False
    while not (terminal or truncated):
        a, aux = act_fn(state)
        frames.append(snap(state, a, aux, info))
        state, _, terminal, truncated, info = env.step(a)
    frames.append(snap(state, None, {}, info))
    return geo, frames, info


def summarize(frames, info):
    sep = []
    for f in frames:
        for it in f["intr"]:
            if it:
                sep.append(math.hypot(f["ego"][0] - it[0], f["ego"][1] - it[1]))
    return dict(reached=bool(info["waypoint_reached"]), time=float(info["flight_time"]),
                intrusions=int(info["intrusion_events"]), restricted=int(info["restricted_area_events"]),
                exits=int(info["sector_exit_events"]), min_sep=min(sep) if sep else float("nan"))


def scan(env, base_fn, base_speed, ac, n, cached=None):
    out = []
    for k, sc in enumerate(ac.eval_scenarios[:n]):
        _, fb, ib = run(env, base_fn, sc, base_speed)
        if cached and k < len(cached):                       # the learned policy's results do not change
            learned = cached[k]["learned"]
        else:
            _, fl, il = run(env, policy_fn(ac), sc, ac.speed)
            learned = summarize(fl, il)
        out.append(dict(k=k, base=summarize(fb, ib), learned=learned))
        print(f"  {k + 1}/{n}", end="\r", flush=True)
    print()
    return out


# ------------------------------------------------------------------------------ drawing
def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def interp(frames, u):
    """State at fractional frame index u (linear between decisions)."""
    n = len(frames)
    u = min(max(u, 0.0), n - 1)
    k = min(int(u), n - 1)
    k1, f = min(k + 1, n - 1), u - k
    a, b = frames[k], frames[k1]
    ego = (a["ego"][0] + f * (b["ego"][0] - a["ego"][0]), a["ego"][1] + f * (b["ego"][1] - a["ego"][1]))
    hdg = a["hdg"] + f * wrap(b["hdg"] - a["hdg"])
    tas = a["tas"] + f * (b["tas"] - a["tas"])
    t = a["t"] + f * (b["t"] - a["t"])
    intr = []
    for ia, ib in zip(a["intr"], b["intr"]):
        if ia is None:
            intr.append(None)
        elif ib is None:
            intr.append(ia)
        else:
            intr.append((ia[0] + f * (ib[0] - ia[0]), ia[1] + f * (ib[1] - ia[1]),
                         ia[2] + f * wrap(ib[2] - ia[2]), ia[3] + f * (ib[3] - ia[3])))
    return k, dict(ego=ego, hdg=hdg, tas=tas, t=t, intr=intr)


def triangle(x, y, hdg, size):
    h = math.radians(hdg)
    d, r = np.array([math.sin(h), math.cos(h)]), np.array([math.cos(h), -math.sin(h)])
    c = np.array([x, y])
    return np.array([c + d * size * 1.5, c - d * size * 0.9 + r * size * 0.8, c - d * size * 0.5,
                     c - d * size * 0.9 - r * size * 0.8])


def cpa_geometry(p, intr):
    """Straight-line closest approach from positions (NM) and velocities: (tcpa s, ego_pred, intr_pred, miss NM)."""
    if intr is None:
        return None
    ve = np.array([math.sin(math.radians(p["hdg"])), math.cos(math.radians(p["hdg"]))]) * p["tas"] / 1852.0
    vi = np.array([math.sin(math.radians(intr[2])), math.cos(math.radians(intr[2]))]) * intr[3] / 1852.0
    rel, vrel = np.array(intr[:2]) - np.array(p["ego"]), vi - ve
    v2 = float(vrel @ vrel)
    if v2 < 1e-12:
        return None
    t = -float(rel @ vrel) / v2
    if t <= 0:
        return None
    ep, ip = np.array(p["ego"]) + ve * t, np.array(intr[:2]) + vi * t
    return t, ep, ip, float(np.hypot(*(ep - ip)))


def readout(fr, p, geo):
    s, m = fr["state"], fr["m"]
    lr = lambda rad: "right" if rad > 0 else "left"
    lines = [f"t = {p['t']:4.0f} s    TAS {p['tas']:5.1f} m/s    heading {p['hdg'] % 360:03.0f} deg",
             f"goal       {s.dist_goal_km:5.1f} km   bearing {math.degrees(s.goal_bearing):+6.1f} deg ({lr(s.goal_bearing)})"]
    if s.obstacle:
        lines.append(f"obstacle   clearance {s.obstacle[0]:5.1f} NM (to circle edge)  bearing {math.degrees(s.obstacle[1]):+6.1f}")
    if s.intruder:
        d, b, _, tcpa, dcpa = s.intruder
        threat = tcpa < GATE_T and abs(dcpa) < GATE_D
        tag = "CONFLICT PREDICTED" if (threat and abs(dcpa) < 5) else ("watch" if threat else "clear")
        lines.append(f"intruder   {d:5.1f} NM   bearing {math.degrees(b):+6.1f} deg")
        lines.append(f"           closest in {tcpa:4.0f} s, miss {abs(dcpa):4.1f} NM on the {lr(dcpa)}   [{tag}]")
    if fr["a"] is not None:
        a = fr["a"] if isinstance(fr["a"], tuple) else (fr["a"], None)
        sp = f"   speed {a[1]:+.2f} ({a[1] * 20 / 3:+.1f} kt)" if a[1] is not None else ""
        lines.append(f"action     heading {a[0]:+.2f} ({a[0] * 45:+.0f} deg){sp}")
    v = fr["aux"].get("v")
    lines.append(f"critic v(s) {v:+.2f}" if v is not None else "critic     (none: not a learned policy)")
    lines.append(f"violations so far: intrusion {m['intrusion_time']:.0f} s | restricted area {m['time_in_restricted_area']:.0f} s")
    return "\n".join(lines)


def dashboard(geo, runs, titles, scen_label, sub=4, save=None, interval=60, snap=None):
    import matplotlib
    if save or snap:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Polygon as MPoly

    blue, grey, orange, red, green, purple = "#0072B2", "#6b7280", "#E69F00", "#D55E00", "#009E73", "#CC79A7"
    cols = [grey, blue]
    fig = plt.figure(figsize=(14, 9.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[5.8, 1.45, 1.6], hspace=0.25, wspace=0.12, left=0.04, right=0.98,
                          top=0.9, bottom=0.06)
    fig.suptitle(f"{scen_label}   |   toy scenario (half scale, one obstacle, one straight-line intruder): not competition scores",
                 fontsize=10.5, color="#444")
    sec = np.array(geo["sector"])
    lim = [sec.min(axis=0) - 6, sec.max(axis=0) + 6]
    maps, texts, art = [], [], []
    for c in range(2):
        ax = fig.add_subplot(gs[0, c])
        ax.set_xlim(lim[0][0], lim[1][0]); ax.set_ylim(lim[0][1], lim[1][1]); ax.set_aspect("equal")
        ax.set_title(titles[c], fontsize=12, color=cols[c], fontweight="bold")
        ax.set_xlabel("east (NM)", fontsize=8); ax.set_ylabel("north (NM)", fontsize=8); ax.tick_params(labelsize=8)
        ax.add_patch(MPoly(sec, closed=True, fill=False, ec="#999", ls="--", lw=1.2))
        for ob in geo["obstacles"]:
            ax.add_patch(MPoly(np.array(ob["poly"]), closed=True, fc=red + "55", ec=red, lw=1.5))
            ax.add_patch(Circle(ob["centre"], ob["radius"], fill=False, ec=red, ls=":", lw=1.1))
        gx, gy = geo["goal"]
        ax.add_patch(Circle((gx, gy), 5 / 1.852, fc=green + "44", ec=green, lw=1.2))
        ax.plot([gx], [gy], marker="*", color=green, ms=11)
        ax.plot(*geo["start"], marker="o", color=grey, ms=4)
        trail_e, = ax.plot([], [], color=cols[c], lw=1.4, alpha=.8)
        trail_i, = ax.plot([], [], color=orange, lw=1.2, alpha=.8)
        ego = MPoly(np.zeros((4, 2)), closed=True, fc=blue, ec="white", lw=1, zorder=5); ax.add_patch(ego)
        intr = MPoly(np.zeros((4, 2)), closed=True, fc=orange, ec="white", lw=1, zorder=5); ax.add_patch(intr)
        bubble = Circle((0, 0), 5.0, fill=False, ec=orange, ls="--", lw=1.4); ax.add_patch(bubble)
        pe, = ax.plot([], [], color=blue, ls=":", lw=1.3)
        pi, = ax.plot([], [], color=orange, ls=":", lw=1.3)
        miss, = ax.plot([], [], color=purple, lw=2.2, marker="o", ms=4)
        misst = ax.text(0, 0, "", color=purple, fontsize=9, fontweight="bold")
        clr, = ax.plot([], [], color=red, lw=1.2, alpha=.8)
        banner = ax.text(0.5, 0.97, "", transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="bold")
        maps.append(ax); art.append(dict(te=trail_e, ti=trail_i, ego=ego, intr=intr, bub=bubble, pe=pe, pi=pi, miss=miss,
                                         misst=misst, clr=clr, banner=banner))
        tx = fig.add_subplot(gs[1, c]); tx.axis("off")
        texts.append(tx.text(0.01, 0.97, "", family="monospace", fontsize=9.2, va="top", transform=tx.transAxes))
    # strip charts
    ax_s, ax_v = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    for c, fr in enumerate(runs):
        t = [f["t"] for f in fr]
        sep = [min([math.hypot(f["ego"][0] - i[0], f["ego"][1] - i[1]) for i in f["intr"] if i] or [np.nan]) for f in fr]
        ax_s.plot(t, sep, color=cols[c], lw=1.6, label=titles[c].split(" (")[0])
        ax_v.plot(t, [f["tas"] for f in fr], color=cols[c], lw=1.6)
    ax_s.axhline(5, color=red, ls="--", lw=1); ax_s.text(2, 5.8, "5 NM separation minimum", color=red, fontsize=8)
    ax_s.set_ylabel("distance to intruder (NM)", fontsize=8); ax_s.set_ylim(0, None); ax_s.tick_params(labelsize=8)
    ax_s.set_xlabel("flight time (s)", fontsize=8)
    ax_v.axhline(165.6, color="#999", ls=":", lw=1); ax_v.text(2, 166.3, "speed cap 165.6 m/s", fontsize=8, color="#777")
    ax_v.axhline(152.4, color="#999", ls=":", lw=1); ax_v.text(2, 153.1, "hold 152.4 m/s", fontsize=8, color="#777")
    ax_v.set_ylabel("airspeed (m/s)", fontsize=8); ax_v.set_ylim(140, 172); ax_v.tick_params(labelsize=8)
    ax_v.set_xlabel("flight time (s)", fontsize=8)
    cur_s = ax_s.axvline(0, color="k", lw=1); cur_v = ax_v.axvline(0, color="k", lw=1)

    n_frames = max(len(r) for r in runs)
    total = (n_frames - 1) * sub + sub * 6          # hold the last frame for a moment

    def draw(step):
        u = step / sub
        tmax = 0
        for c, fr in enumerate(runs):
            k, p = interp(fr, u)
            a = art[c]
            kk = int(min(u, len(fr) - 1))
            ex = [f["ego"][0] for f in fr[:kk + 1]] + [p["ego"][0]]; ey = [f["ego"][1] for f in fr[:kk + 1]] + [p["ego"][1]]
            a["te"].set_data(ex, ey)
            a["ego"].set_xy(triangle(*p["ego"], p["hdg"], 2.6))
            i0 = p["intr"][0] if p["intr"] else None
            a["ti"].set_data([f["intr"][0][0] for f in fr[:kk + 1] if f["intr"][0]] + ([i0[0]] if i0 else []),
                             [f["intr"][0][1] for f in fr[:kk + 1] if f["intr"][0]] + ([i0[1]] if i0 else []))
            if i0:
                a["intr"].set_xy(triangle(i0[0], i0[1], i0[2], 2.6)); a["bub"].center = (i0[0], i0[1])
                sep = math.hypot(p["ego"][0] - i0[0], p["ego"][1] - i0[1])
                a["bub"].set_edgecolor(red if sep < 5 else orange); a["bub"].set_linewidth(2.4 if sep < 5 else 1.4)
                a["intr"].set_visible(True); a["bub"].set_visible(True)
            else:
                a["intr"].set_visible(False); a["bub"].set_visible(False)
            cp = cpa_geometry(p, i0)
            if cp and cp[0] <= GATE_T:
                t_, ep, ip, ms = cp
                a["pe"].set_data([p["ego"][0], ep[0]], [p["ego"][1], ep[1]]); a["pi"].set_data([i0[0], ip[0]], [i0[1], ip[1]])
                a["miss"].set_data([ep[0], ip[0]], [ep[1], ip[1]])
                a["misst"].set_position(((ep[0] + ip[0]) / 2 + 2, (ep[1] + ip[1]) / 2 + 2)); a["misst"].set_text(f"miss {ms:.1f} NM")
            else:
                for key in ("pe", "pi", "miss"):
                    a[key].set_data([], [])
                a["misst"].set_text("")
            s = fr[k]["state"]
            if s.obstacle and geo["obstacles"]:
                ob = geo["obstacles"][0]; v = np.array(ob["centre"]) - np.array(p["ego"]); d = np.hypot(*v)
                edge = np.array(p["ego"]) + v / d * max(d - ob["radius"], 0)
                a["clr"].set_data([p["ego"][0], edge[0]], [p["ego"][1], edge[1]])
            texts[c].set_text(readout(fr[k], p, geo))
            done = u >= len(fr) - 1
            m = fr[-1]["m"]
            a["banner"].set_text((f"GOAL REACHED at {fr[-1]['t']:.0f} s | intrusions {m['intrusion_events']:.0f} | "
                                  f"restricted entries {m['restricted_area_events']:.0f}") if done else "")
            a["banner"].set_color(green if (done and m["intrusion_events"] == 0 and m["restricted_area_events"] == 0) else red)
            tmax = max(tmax, p["t"])
        cur_s.set_xdata([tmax, tmax]); cur_v.set_xdata([tmax, tmax])
        return []

    draw(0)
    if snap:                                           # (png path, fraction of the replay): one still frame
        draw(int(snap[1] * (n_frames - 1) * sub))
        fig.savefig(snap[0], dpi=80)
        print("wrote", snap[0])
        plt.close(fig)
        return fig, None
    anim = FuncAnimation(fig, draw, frames=total, interval=interval, blit=False, repeat=True)
    if save:
        anim.save(save, writer=PillowWriter(fps=int(1000 / interval)), dpi=72)
        print("wrote", save)
        plt.close(fig)
    return fig, anim


def slide(results=None, save=None):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 8.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(.03, .955, "A policy-gradient agent on the competition environment: what we changed", fontsize=19, fontweight="bold", va="top")
    ax.text(.03, .905, "We keep the environment and the scoring untouched and edit only the three hooks, plus the learner that sits outside them.",
            fontsize=11.5, color="#444", va="top")
    colx = [.03, .35, .67]
    heads = ["Observation  (_get_obs)", "Action  (_get_action)", "Reward  (_get_reward)"]
    shipped = ["131 numbers: goal drift + distance, airspeed,\n10 intruder slots x 7, 5 obstacles x 4,\n12 sector points + inside flag.\nObstacles are a centroid + enclosing radius.",
               "heading change +-45 deg and speed change\n+-6.67 kt, per 10 s decision.",
               "+1 reaching the goal, -0.01 x |drift| (rad) per step,\n-1 per second in intrusion / restricted area / outside\nthe sector (from the scored metrics)."]
    ours = ["A few physical quantities, all relative to own heading:\n goal distance + bearing;\n obstacle clearance to the circle edge + bearing;\n nearest intruder distance, bearing, aspect;\n straight-line closest approach: time + signed miss distance;\n own airspeed.   Tile-coded (8 tilings), linear in features.",
            "Same two channels. One Gaussian head each:\n mean and log-sigma are linear in the tile features.\n The speed head is new this week. BlueSky allows\n 67-163 m/s calibrated airspeed (165.6 m/s true at the top).",
            "-1 per second flown (flight time is a scored metric)\n+ potential-based progress toward the goal\n+ goal bonus\n- penalties per violation event / second, read from\n  the scored metrics (so reward follows the score)."]
    for x, h, s_, o in zip(colx, heads, shipped, ours):
        ax.text(x, .84, h, fontsize=14, fontweight="bold", color="#0072B2", va="top")
        ax.text(x, .795, "shipped", fontsize=9.5, color="#888", va="top")
        ax.text(x, .775, s_, fontsize=10.2, color="#555", va="top", linespacing=1.35)
        ax.text(x, .625, "ours", fontsize=9.5, color="#888", va="top")
        ax.text(x, .605, o, fontsize=10.2, va="top", linespacing=1.35)
    ax.text(.03, .405, "Learner (outside the hooks): one-step actor-critic, Sutton & Barto ch. 13: the critic's TD error  delta = R + v(S') - v(S)  "
            "drives both the policy-mean and critic updates.", fontsize=10.8, va="top", wrap=True)
    r = results or {}
    rows = [("Go-to-goal pursuit (hand-coded)",) + tuple(r.get("pursuit", ("-",) * 5)),
            ("Heading only (learned)", "100%", "2%", "7%", "0%", "481 s"),
            ("Heading + speed (learned)", "100%", "8%", "5%", "0%", "447 s")]
    ax.text(.03, .335, "Result on 100 fixed toy scenarios (greedy policy, one training seed)", fontsize=12.5, fontweight="bold", va="top")
    hdr = ["", "goal reached", "restricted entry", "intrusion", "sector exit", "mean time"]
    xs = [.03, .30, .42, .55, .66, .77]
    for x, h in zip(xs, hdr):
        ax.text(x, .295, h, fontsize=10, color="#666", fontweight="bold", va="top")
    for i, row in enumerate(rows):
        for x, v in zip(xs, row):
            ax.text(x, .262 - i * .037, v, fontsize=10.8, va="top")
    ax.text(.03, .13, "Honest limits: half-scale toy scenarios (competition: 5 obstacles, 10 intruders), 100 scenarios per row so the error bars are wide, "
            "single seed.\nSpeed gave ~6% shorter flights by speeding up everywhere, not yet by timing crossings; that needs denser traffic.\n"
            "Next: multi-intruder stage, pick the most urgent intruder by closest approach, then the multi-agent track.",
            fontsize=10.2, color="#444", va="top", linespacing=1.5)
    if save:
        fig.savefig(save, dpi=110)
        print("wrote", save)
    return fig


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=None, help="index 0-99 into the fixed evaluation scenarios (default 28: "
                    "pursuit flies into the obstacle and the intruder)")
    ap.add_argument("--policy", default=LEARNED)
    ap.add_argument("--baseline", default="pursuit", help="'pursuit' (hand-coded go-to-goal) or a checkpoint path")
    ap.add_argument("--gate", type=float, nargs=2, default=None)
    ap.add_argument("--find", action="store_true")
    ap.add_argument("--n", type=int, default=100, help="scenarios scanned by --find")
    ap.add_argument("--save", help="write the animation to this .gif instead of opening a window")
    ap.add_argument("--save-slide", help="write the strategy slide to this .png")
    ap.add_argument("--no-slide", action="store_true")
    ap.add_argument("--png", help="write one still frame of the replay to this .png (with --at)")
    ap.add_argument("--at", type=float, default=0.5, help="fraction of the replay for --png")
    args = ap.parse_args()

    env = NavEnv(stage=3, scale=0.5, seed=0)
    ac = load_policy(env, args.policy, tuple(args.gate) if args.gate else None)
    if args.baseline == "pursuit":
        base_fn, base_speed, base_title = pursuit_fn(False), False, "Go-to-goal pursuit (hand-coded, ignores traffic)"
    else:
        bac = load_policy(env, args.baseline)
        base_fn, base_speed, base_title = policy_fn(bac), bac.speed, "Baseline policy (" + os.path.basename(args.baseline) + ")"

    if args.find:
        path = os.path.join(HERE, "figures", "_demo_scan.json")
        cached = json.load(open(path)) if (os.path.exists(path) and args.policy == LEARNED) else None
        rows = scan(env, base_fn, base_speed, ac, args.n, cached)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(rows, open(path, "w"), indent=1)

        def agg(key):
            s = [r[key] for r in rows]
            reached = [x for x in s if x["reached"]]
            return (f"{np.mean([x['reached'] for x in s]):.0%}", f"{np.mean([x['restricted'] > 0 for x in s]):.0%}",
                    f"{np.mean([x['intrusions'] > 0 for x in s]):.0%}", f"{np.mean([x['exits'] > 0 for x in s]):.0%}",
                    f"{np.mean([x['time'] for x in reached]):.0f} s")
        print("baseline:", agg("base")); print("learned: ", agg("learned"))
        good = [r for r in rows if (r["base"]["intrusions"] > 0 or r["base"]["restricted"] > 0)
                and r["learned"]["intrusions"] == 0 and r["learned"]["restricted"] == 0 and r["learned"]["reached"]]
        print("candidates (baseline violates, learned is clean):")
        for r in sorted(good, key=lambda r: r["base"]["min_sep"])[:12]:
            print(f"  scenario {r['k']:3d}: baseline min sep {r['base']['min_sep']:5.1f} NM, restricted {r['base']['restricted']}, "
                  f"time {r['base']['time']:.0f} s | learned min sep {r['learned']['min_sep']:5.1f} NM, time {r['learned']['time']:.0f} s")
        return

    k = 28 if args.scenario is None else args.scenario
    scenario = ac.eval_scenarios[k]
    print(f"recording scenario {k} with the baseline and the learned policy ...", flush=True)
    geo, fb, ib = run(env, base_fn, scenario, base_speed)
    _, fl, il = run(env, policy_fn(ac), scenario, ac.speed)
    sb, sl = summarize(fb, ib), summarize(fl, il)
    print(f"  baseline: {sb}\n  learned : {sl}")
    results = None
    sp = os.path.join(HERE, "figures", "_demo_scan.json")
    if os.path.exists(sp) and args.baseline == "pursuit":
        rows = json.load(open(sp)); s = [r["base"] for r in rows]; rr = [x for x in s if x["reached"]]
        results = dict(pursuit=(f"{np.mean([x['reached'] for x in s]):.0%}", f"{np.mean([x['restricted'] > 0 for x in s]):.0%}",
                                f"{np.mean([x['intrusions'] > 0 for x in s]):.0%}", f"{np.mean([x['exits'] > 0 for x in s]):.0%}",
                                f"{np.mean([x['time'] for x in rr]):.0f} s"))
    label = f"scenario {k}"
    if args.save_slide or (not args.no_slide and not args.save and not args.png):
        slide(results, args.save_slide)
    fig, anim = dashboard(geo, [fb, fl], [base_title, "Learned policy (heading + speed)" if ac.speed else "Learned policy"],
                          label, save=args.save, snap=(args.png, args.at) if args.png else None)
    if not args.save and not args.png:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
