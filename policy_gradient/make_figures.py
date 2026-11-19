"""Make the figures used in policy_gradient/THEORY.md.

    python -m policy_gradient.make_figures              # all figures (re-uses cached rollouts)
    python -m policy_gradient.make_figures --recollect  # re-fly the trained policies first

Needs matplotlib. Rollouts of the trained policies are collected in one subprocess per stage
(BlueSky allows one env per process) and cached in policy_gradient/figures/_rollouts_stage<N>.json.
The figures only use saved checkpoints / histories / eval outputs, so nothing is retrained.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "checkpoints")
FIG = os.path.join(HERE, "figures")
CENTER = (52.0, 4.0)

# which trained policy is shown for each stage
RUNS = {1: ("stage1.npz", False), 2: ("stage2_ep5.npz", False), 3: ("stage3_cpa.npz", True)}

# Okabe-Ito palette (colour-blind safe)
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7",
         sky="#56B4E9", yellow="#F0E442", grey="#666666", light="#DDDDDD", ink="#222222")


def to_nm(lat, lon):
    """(lat, lon) -> (east, north) in nautical miles from the sector centre."""
    return ((lon - CENTER[1]) * 60.0 * math.cos(math.radians(CENTER[0])), (lat - CENTER[0]) * 60.0)


# ======================================================================== rollout collection
def geometry(scenario):
    ag = scenario.agents[0]
    return dict(
        sector=[to_nm(*p) for p in scenario.sector],
        obstacles=[[to_nm(*p) for p in o.vertices] for o in scenario.obstacles],
        routes=[[to_nm(*p) for p in r.waypoints] for r in scenario.intruder_routes],
        start=to_nm(*ag.start), goal=to_nm(*ag.goal), heading=ag.heading)


def collect(stage):
    import bluesky as bs
    from policy_gradient.nav_env import NavEnv
    from policy_gradient.actor_critic import ActorCritic

    ck_name, cpa = RUNS[stage]
    env = NavEnv(stage=stage)
    ac = ActorCritic(env, stage, n_eval=4, cpa=cpa)
    ck = np.load(os.path.join(CK, ck_name))
    n = ck["w"].shape[0]
    ac.w[:n], ac.theta_mu[:n], ac.theta_sigma[:n] = ck["w"], ck["theta_mu"], ck["theta_sigma"]

    def pos(ac_id):
        i = bs.traf.id2idx(ac_id)
        return to_nm(float(bs.traf.lat[i]), float(bs.traf.lon[i]))

    def fly(label, scenario, policy):
        state = env.reset(scenario)
        geom = geometry(env.env.scenario)
        agent, intr = [pos("KL001")], [pos("AC1")] if stage >= 3 else []
        done, info = False, env.info
        while not done:
            state, _, term, trunc, info = env.step(policy(state))
            agent.append(pos("KL001"))
            if stage >= 3:
                intr.append(pos("AC1"))
            done = term or trunc
        keep = ("waypoint_reached", "flight_time", "intrusion_events", "restricted_area_events", "sector_exit_events")
        return dict(label=label, geom=geom, agent=agent, intruder=intr, step_s=env.max_step_seconds,
                    info={k: float(info[k]) for k in keep})

    def learned(s):
        return ac.act(s, greedy=True)

    def straight_at_goal(s):                      # ignores obstacles and intruders entirely
        return 2.0 * s.goal_bearing

    out = [fly("steer at goal only (no avoidance)", "canonical", straight_at_goal),
           fly("learned: canonical layout", "canonical", learned),
           fly("learned: random scenario A", ac.eval_scenarios[0], learned),
           fly("learned: random scenario B", ac.eval_scenarios[1], learned)]
    with open(os.path.join(FIG, f"_rollouts_stage{stage}.json"), "w") as f:
        json.dump(out, f)
    env.close()


def ensure_rollouts(recollect):
    for stage in (1, 2, 3):
        path = os.path.join(FIG, f"_rollouts_stage{stage}.json")
        if recollect or not os.path.exists(path):
            print(f"collecting stage {stage} rollouts ...", flush=True)
            env = dict(os.environ, SDL_VIDEODRIVER="dummy", PYTHONPATH=os.path.dirname(HERE))
            subprocess.run([sys.executable, "-m", "policy_gradient.make_figures", "--collect", str(stage)],
                           check=True, env=env, cwd=os.path.dirname(HERE), stdout=subprocess.DEVNULL)


# ============================================================================== drawing helpers
def style(plt):
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["grey"],
        "axes.labelcolor": C["ink"], "xtick.color": C["ink"], "ytick.color": C["ink"],
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 150,
        "savefig.bbox": "tight", "legend.frameon": False})


def draw_layout(ax, g, route_color=C["orange"]):
    from matplotlib.patches import Polygon, Circle
    ax.add_patch(Polygon(g["sector"], closed=True, fill=False, ec=C["ink"], lw=1.4))
    for o in g["obstacles"]:
        ax.add_patch(Polygon(o, closed=True, fc=C["grey"], ec="none", alpha=0.75))
    for r in g["routes"]:
        ax.plot(*zip(*r), color=route_color, lw=1.2, ls=":", zorder=1)
        (x0, y0), (x1, y1) = r[0], r[-1]
        ax.annotate("", xy=(x1, y1), xytext=(x0 + 0.8 * (x1 - x0), y0 + 0.8 * (y1 - y0)),
                    arrowprops=dict(arrowstyle="-|>", color=route_color, lw=1.2), zorder=1)
    ax.add_patch(Circle(g["goal"], 5 / 1.852, fc=C["green"], ec="none", alpha=0.9, zorder=2))   # 5 km capture radius
    ax.plot(*g["start"], "o", color=C["blue"], ms=6, zorder=5)
    h = math.radians(g["heading"])                       # heading: degrees clockwise from north
    ax.annotate("", xy=(g["start"][0] + 7 * math.sin(h), g["start"][1] + 7 * math.cos(h)), xytext=g["start"],
                arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=1.5), zorder=5)
    xs = [p[0] for p in g["sector"]] + [p[0] for r in g["routes"] for p in r]
    ys = [p[1] for p in g["sector"]] + [p[1] for r in g["routes"] for p in r]
    pad = 4
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def sep_nm(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ===================================================================================== figures
def fig_pipeline(plt):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12.5, 5.9))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.9); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=color, ec="none", alpha=0.16))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc="none", ec=color, lw=1.6))
        ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontweight="bold", fontsize=10.5, color=C["ink"])
        ax.text(x + w / 2, y + h - 0.68, body, ha="center", va="top", fontsize=9, color=C["ink"], linespacing=1.45)

    def arrow(p, q, color=C["ink"]):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

    def label(x, y, text, color=C["ink"], ha="center", size=8.8):
        ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
                bbox=dict(fc="white", ec="none", pad=1.2))

    ax.text(0.1, 5.75, "WHAT WE LEARN  (actor_critic.py)", fontsize=9.5, color=C["grey"], fontweight="bold")
    ax.text(6.4, 5.75, "THE WORLD  (nothing here is learned)", fontsize=9.5, color=C["grey"], fontweight="bold")

    box(0.1, 3.7, 3.0, 1.85, "Features  x(s)", "tile-coded groups:\ngoal, obstacle, intruder\n(8 active tiles per group)", C["sky"])
    box(0.1, 0.6, 3.0, 1.85, "Actor + critic", "policy  N( mu(s), sigma(s)^2 )\nmu = theta_mu . x(s)\nvalue  v(s) = w . x(s)", C["blue"])
    box(4.5, 0.35, 2.3, 1.35, "TD error", "delta = R + v(S') - v(S)\n(R comes from the world)", C["purple"])
    box(7.0, 3.7, 2.6, 1.85, "NavEnv adapter", "action: heading only\nstate: a few numbers\nreward: shaped\n(nav_env.py)", C["orange"])
    box(7.0, 0.6, 2.6, 1.85, "CompetitionEnv", "scenario + scored metrics\n(competition code,\nnot modified)", C["green"])
    box(10.3, 0.6, 2.1, 1.85, "BlueSky", "A320 flight model:\nturn lag, speed,\nthe intruder", C["grey"])

    # perception: world -> features
    arrow((7.0, 4.9), (3.1, 4.9))
    label(5.05, 4.9, "state S' and reward R")
    arrow((1.6, 3.7), (1.6, 2.5))
    label(1.6, 3.1, "x(s)", ha="center")
    # action: actor -> world
    arrow((3.1, 2.0), (7.0, 4.0))
    label(4.95, 3.15, "sample action  a ~ N(mu, sigma^2)")
    # inside the world
    arrow((7.7, 3.7), (7.7, 2.5)); label(7.7, 3.1, "[a, 0]", ha="center")
    arrow((8.9, 2.5), (8.9, 3.7)); label(8.9, 3.1, "obs + metrics", ha="center")
    arrow((9.6, 1.9), (10.3, 1.9)); label(9.95, 2.15, "HDG")
    arrow((10.3, 1.2), (9.6, 1.2)); label(9.95, 0.95, "state")
    # learning signal
    arrow((4.5, 1.0), (3.1, 1.0), color=C["purple"])
    label(3.8, 1.3, "update theta, w", color=C["purple"], size=8.5)
    ax.text(6.25, 0.08, "one decision every 10 simulated seconds; about 50-100 decisions per episode at scale 0.5",
            ha="center", fontsize=9, color=C["grey"])
    fig.savefig(os.path.join(FIG, "fig0_pipeline.png"))
    plt.close(fig)


def fig_scenarios(plt):
    sys.path.insert(0, os.path.dirname(HERE))
    from policy_gradient.scenarios import canonical_scenario, StageGenerator
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
    titles = ["Stage 1: goal only", "Stage 2: + one obstacle", "Stage 3: + one crossing intruder"]
    for k, stage in enumerate((1, 2, 3)):
        for row, sc in enumerate((canonical_scenario(stage, 0.5),
                                  StageGenerator(stage, fixed_prob=0.0, scale=0.5).generate(np.random.default_rng(3)))):
            ax = axes[row][k]
            draw_layout(ax, geometry(sc))
            if row == 0:
                ax.set_title(titles[k])
            ax.text(0.02, 0.02, "canonical layout" if row == 0 else "a random training scenario",
                    transform=ax.transAxes, fontsize=8.5, color=C["grey"])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axes[1][1].legend(handles=[
        Line2D([], [], marker="o", color=C["blue"], ls="", label="start (arrow = initial heading)"),
        Patch(fc=C["green"], alpha=0.9, label="goal (captured within 5 km)"),
        Patch(fc=C["grey"], alpha=0.75, label="restricted area"),
        Line2D([], [], color=C["orange"], ls=":", label="intruder's planned route (arrow = direction)"),
        Line2D([], [], color=C["ink"], lw=1.4, label="sector boundary")],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9)
    fig.suptitle("The three staged scenarios (scale 0.5: a 60 NM sector, half the competition size)",
                 y=0.96, fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(FIG, "fig1_scenarios.png"))
    plt.close(fig)


def fig_trajectories(plt):
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for r, stage in enumerate((1, 2, 3)):
        with open(os.path.join(FIG, f"_rollouts_stage{stage}.json")) as f:
            cases = json.load(f)
        for c, case in enumerate(cases):
            ax = axes[r][c]
            draw_layout(ax, case["geom"])
            agent = np.array(case["agent"])
            n = len(agent)
            every = max(1, int(round(60 / case["step_s"])))          # a dot every 60 s
            ax.plot(agent[:, 0], agent[:, 1], color=C["blue"], lw=1.8, zorder=4)
            ax.plot(agent[::every, 0], agent[::every, 1], "o", color=C["blue"], ms=3, zorder=4)
            title_extra = ""
            if case["intruder"]:
                intr = np.array(case["intruder"])
                m = min(len(intr), n)
                ax.plot(intr[:m, 0], intr[:m, 1], color=C["orange"], lw=1.6, zorder=3)
                ax.plot(intr[:m:every, 0], intr[:m:every, 1], "o", color=C["orange"], ms=3, zorder=3)
                seps = [sep_nm(agent[i], intr[i]) for i in range(m)]
                i = int(np.argmin(seps))
                ax.plot([agent[i, 0], intr[i, 0]], [agent[i, 1], intr[i, 1]], color=C["red"], lw=1.2, zorder=6)
                ax.text(0.5, -0.03, f"closest approach {seps[i]:.1f} NM",
                        transform=ax.transAxes, fontsize=8.3, ha="center", va="top",
                        color=C["red"] if seps[i] < 5 else C["grey"])
            info = case["info"]
            verdict = ("reached goal" if info["waypoint_reached"] else "MISSED goal")
            flags = []
            if info["restricted_area_events"]:
                flags.append(f"{int(info['restricted_area_events'])}x restricted")
            if info["intrusion_events"]:
                flags.append(f"{int(info['intrusion_events'])}x intrusion")
            if info["sector_exit_events"]:
                flags.append(f"{int(info['sector_exit_events'])}x sector exit")
            ax.set_title(f"{case['label']}\n{verdict} in {info['flight_time']:.0f} s"
                         + (" | " + ", ".join(flags) if flags else " | clean"), fontsize=9.3,
                         color=C["red"] if flags else C["ink"])
            if c == 0:
                ax.set_ylabel(f"Stage {stage}", fontsize=12, fontweight="bold")
                ax.yaxis.set_label_position("left")
                ax.set_yticks([]);
    fig.suptitle("Trained policies in action (blue = our aircraft, orange = intruder; dots every 60 s)",
                 y=0.93, fontsize=12.5, fontweight="bold")
    fig.savefig(os.path.join(FIG, "fig2_trajectories.png"))
    plt.close(fig)


def fig_learning(plt):
    runs = [("stage 1", "stage1_history.json", C["grey"], "-"),
            ("stage 2, penalty 0.5", "stage2_history.json", C["sky"], "-"),
            ("stage 2, penalty 5", "stage2_ep5_history.json", C["blue"], "-"),
            ("stage 3, old features", "stage3_history.json", C["orange"], "-"),
            ("stage 3, + CPA features", "stage3_cpa_history.json", C["green"], "-"),
            ("stage 3, CPA warm start", "stage3_cpa_warm_history.json", C["purple"], "--")]
    panels = [("mean_time", "mean flight time of successful episodes (s)", 1.0),
              ("restricted", "scenarios with a restricted-area entry (%)", 100.0),
              ("intrusion", "scenarios with a loss of separation (%)", 100.0),
              ("sigma", "policy sigma along the canonical drive", 1.0)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    for ax, (key, label, mult) in zip(axes.ravel(), panels):
        for name, fn, col, ls in runs:
            h = json.load(open(os.path.join(CK, fn)))
            ax.plot([e["episode"] for e in h], [e[key] * mult for e in h], color=col, ls=ls, lw=1.7, label=name)
        ax.set_xlabel("training episodes")
        ax.set_title(label, fontsize=10)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Learning curves (each point = greedy evaluation on 20 fixed random scenarios, so +/- 5 points is one scenario)",
                 y=0.995, fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(FIG, "fig3_learning.png"))
    plt.close(fig)


def parse_eval(fn):
    out = {}
    for line in open(os.path.join(CK, fn), encoding="utf-8-sig"):
        m = re.match(r"\s*(goal reached|restricted entry|intrusion|sector exit)\s+(\d+)%\s+\(95% CI\s+(\d+)%-\s*(\d+)%\)", line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return out


def fig_final(plt):
    groups = [("stage 2 (penalty 5)", "eval_s2.txt", C["blue"]),
              ("stage 3, old features", "eval_s3.txt", C["orange"]),
              ("stage 3, + CPA, scratch", "eval_s3_cpa.txt", C["green"]),
              ("stage 3, + CPA, warm", "eval_s3_cpa_warm.txt", C["purple"])]
    cats = ["restricted entry", "intrusion", "sector exit"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    w = 0.2
    for gi, (name, fn, col) in enumerate(groups):
        d = parse_eval(fn)
        xs = np.arange(len(cats)) + (gi - 1.5) * w
        vals = [d[c][0] for c in cats]
        lo = [d[c][0] - d[c][1] for c in cats]
        hi = [d[c][2] - d[c][0] for c in cats]
        ax.bar(xs, vals, w * 0.92, color=col, label=name, yerr=[lo, hi], error_kw=dict(ecolor=C["ink"], lw=1, capsize=2))
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.8, f"{v}%", ha="center", fontsize=8.5, color=C["ink"])
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(["entered a restricted area", "lost separation (< 5 NM)", "left the sector"])
    ax.set_ylabel("share of 100 random scenarios (%)")
    ax.set_ylim(0, 24)
    ax.legend(fontsize=9)
    ax.set_title("Final evaluation, 100 random scenarios, greedy policy (bars: 95% interval; goal reached: 100% for all)")
    fig.savefig(os.path.join(FIG, "fig4_final_eval.png"))
    plt.close(fig)


def fig_policy(plt):
    sys.path.insert(0, os.path.dirname(HERE))
    from policy_gradient.actor_critic import Features
    from policy_gradient.nav_env import State

    def load(stage):
        name, cpa = RUNS[stage]
        ck = np.load(os.path.join(CK, name))
        feat = Features(stage, cpa=cpa)

        def mu_sigma(s):
            idx = feat.indices(s)
            mu = ck["theta_mu"][idx].sum()
            sig = math.exp(min(max(ck["theta_sigma"][idx].sum(), math.log(0.05)), 0.0))
            return mu, sig
        return mu_sigma

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1.05, 1.05]))

    # (a) stage 1: steering law
    ms = load(1)
    ax = axes[0]
    b = np.linspace(-math.pi, math.pi, 361)
    for dist, col in ((15, C["sky"]), (35, C["blue"]), (60, C["purple"])):
        ax.plot(np.degrees(b), [ms(State(dist, x))[0] for x in b], color=col, lw=1.8, label=f"goal {dist} km away")
    mus = np.array([ms(State(35, x))[0] for x in b]); sg = np.array([ms(State(35, x))[1] for x in b])
    ax.fill_between(np.degrees(b), mus - sg, mus + sg, color=C["blue"], alpha=0.12, lw=0)
    ax.axhline(0, color=C["grey"], lw=0.6)
    ax.set_ylim(-1.8, 1.8)
    ax.axhspan(1, 1.8, color=C["light"], alpha=0.6, lw=0); ax.axhspan(-1.8, -1, color=C["light"], alpha=0.6, lw=0)
    ax.set_xlabel("bearing of the goal relative to our heading (deg, + = goal on our right)")
    ax.set_ylabel("mean action mu(s)")
    ax.set_title("(a) Stage 1: learned steering law")
    ax.text(-175, 1.5, "above 0: turn right (grey = beyond the +/-1 limit)", fontsize=8.2, color=C["grey"])
    ax.text(-175, -1.72, "below 0: turn left", fontsize=8.2, color=C["grey"])
    ax.legend(fontsize=8.3, loc="lower right", bbox_to_anchor=(1.0, 0.1))

    # (b) stage 2: obstacle avoidance map, goal straight ahead
    ms = load(2)
    ax = axes[1]
    bear = np.linspace(-math.pi / 2, math.pi / 2, 121)
    clr = np.linspace(-5, 40, 91)
    Z = np.array([[ms(State(100, 0.0, obstacle=(c, bb)))[0] for bb in bear] for c in clr])
    im = ax.pcolormesh(np.degrees(bear), clr, Z, cmap="RdBu_r", vmin=-1.2, vmax=1.2, shading="auto")
    ax.set_xlabel("bearing of the obstacle (deg, + = on our right)")
    ax.set_ylabel("clearance to its edge (NM)")
    ax.set_title("(b) Stage 2: mu(s) with the goal dead ahead")
    cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.set_label("mu: red = turn right, blue = turn left")

    # (c) stage 3: conflict map
    ms = load(3)
    ax = axes[2]
    tc = np.linspace(0, 400, 81)
    dc = np.linspace(-15, 15, 91)
    Z = np.array([[ms(State(100, 0.0, obstacle=None, intruder=(25.0, 0.6, 0.0, t, d)))[0] for t in tc] for d in dc])
    im = ax.pcolormesh(tc, dc, Z, cmap="RdBu_r", vmin=-1.2, vmax=1.2, shading="auto")
    ax.axhspan(-5, 5, color="black", alpha=0.0)
    ax.axhline(5, color=C["ink"], lw=0.8, ls="--"); ax.axhline(-5, color=C["ink"], lw=0.8, ls="--")
    ax.text(392, 5.4, "5 NM separation", ha="right", fontsize=8.3)
    ax.set_xlabel("time to closest approach (s)")
    ax.set_ylabel("predicted miss distance (NM, + = passes on our right)")
    ax.set_title("(c) Stage 3: mu(s) vs time and miss distance\n(other intruder inputs held at one geometry)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.set_label("mu: red = turn right, blue = turn left")
    fig.suptitle("What the trained policies learned (greedy mean action, 1.0 = full 45 degree heading command)",
                 y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_learned_policy.png"))
    plt.close(fig)


def fig_representation(plt):
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    P = (0.43, 0.52)
    cols = [C["blue"], C["orange"], C["green"]]
    w = 0.25
    for k, col in enumerate(cols):
        off = (k * 1 / 3 * w, k * 3 / 3 * w * 0.5 % w)               # shifted grids (asymmetric offsets)
        for i in range(-1, 6):
            ax.axvline(off[0] + (i - 1) * w + w * 0.0, color=col, lw=0.7, alpha=0.5)
            ax.axhline(off[1] + (i - 1) * w, color=col, lw=0.7, alpha=0.5)
        ix = math.floor((P[0] - off[0]) / w); iy = math.floor((P[1] - off[1]) / w)
        ax.add_patch(Rectangle((off[0] + ix * w, off[1] + iy * w), w, w, fc=col, alpha=0.22, ec=col, lw=1.6))
    ax.plot(*P, "o", color=C["ink"], ms=7, zorder=5)
    ax.text(P[0] + 0.02, P[1] + 0.03, "state s", fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("input 1 (e.g. distance to goal)"); ax.set_ylabel("input 2 (e.g. goal bearing)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Tile coding: 3 shifted grids shown (we use 8)\none active tile per grid; the overlap is finer than one tile")

    ax = axes[1]
    a = np.linspace(-2.2, 2.2, 600)
    mu = 0.45
    for sg, col, lab in ((0.5, C["sky"], "early training, sigma = 0.5: explores"),
                         (0.1, C["blue"], "later, sigma = 0.1: nearly deterministic")):
        ax.plot(a, np.exp(-0.5 * ((a - mu) / sg) ** 2) / (sg * math.sqrt(2 * math.pi)), color=col, lw=2, label=lab)
    ax.axvline(mu, color=C["grey"], lw=0.8, ls=":")
    ax.text(mu - 0.05, 4.35, "mean mu(s)", fontsize=9, color=C["grey"], ha="right")
    ax.set_ylim(-0.2, 5.4)
    ax.axvspan(-2.2, -1, color=C["light"], alpha=0.6); ax.axvspan(1, 2.2, color=C["light"], alpha=0.6)
    ax.text(1.6, 3.0, "action is clipped\nto [-1, 1]", ha="center", fontsize=8.8, color=C["grey"])
    ax.text(-1.6, 3.0, "action is clipped\nto [-1, 1]", ha="center", fontsize=8.8, color=C["grey"])
    ax.set_xlabel("action a  (heading change as a fraction of 45 degrees; + = right)")
    ax.set_ylabel("probability density  pi(a | s)")
    ax.set_title("The policy is a bell curve over the action\nmu and sigma are both learned functions of the state")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    fig.savefig(os.path.join(FIG, "fig6_representation.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=int, help="(internal) fly the trained policy of this stage and cache it")
    ap.add_argument("--recollect", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    if args.collect:
        collect(args.collect)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    ensure_rollouts(args.recollect)
    for fn in (fig_pipeline, fig_scenarios, fig_trajectories, fig_learning, fig_final, fig_policy, fig_representation):
        print("making", fn.__name__, flush=True)
        fn(plt)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
