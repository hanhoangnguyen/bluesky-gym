"""Train / watch the one-step actor-critic on the staged CompetitionEnv scenarios.

Run from the repo root (BlueSky allows only one env per process, so one stage per run):

    python -m policy_gradient.train --stage 1 --episodes 3000
    python -m policy_gradient.train --stage 2 --episodes 6000 --load policy_gradient/checkpoints/stage1.npz
    python -m policy_gradient.train --stage 2 --watch            # render the saved policy (needs a display)

Stages: 1 goal only | 2 + one obstacle | 3 + one crossing intruder.
"""

import argparse
import json
import os

import numpy as np

from policy_gradient.nav_env import NavEnv
from policy_gradient.actor_critic import ActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=20, help="random scenarios per evaluation")
    ap.add_argument("--scale", type=float, default=0.5, help="geometry scale, 1.0 = competition size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event-penalty", type=float, default=0.5,
                    help="reward cost of each restricted-area entry / sector exit / intrusion")
    ap.add_argument("--lam", type=float, default=0.01, help="reward cost per second spent in violation")
    ap.add_argument("--load", help="start from this checkpoint (e.g. the previous stage's)")
    ap.add_argument("--out", help="checkpoint to write (default policy_gradient/checkpoints/stage<N>.npz)")
    ap.add_argument("--watch", action="store_true", help="no training: render the saved policy")
    ap.add_argument("--evaluate", action="store_true",
                    help="no training: greedy evaluation of the saved policy on --eval-n scenarios")
    args = ap.parse_args()

    out = args.out or f"policy_gradient/checkpoints/stage{args.stage}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    env = NavEnv(stage=args.stage, scale=args.scale, seed=args.seed,
                 event_penalty=args.event_penalty, lam=args.lam,
                 render_mode="human" if args.watch else None)
    ac = ActorCritic(env, args.stage, seed=args.seed, n_eval=args.eval_n)

    src = args.load or (out if (args.watch or args.evaluate) else None)
    if src:
        ck = np.load(src)
        n_ck = ck["w"].shape[0]
        if n_ck > ac.w.shape[0]:
            raise SystemExit(f"{src} has {n_ck} features, stage {args.stage} only {ac.w.shape[0]}.")
        # feature groups are appended stage by stage (goal group first), so an earlier stage's
        # checkpoint fills the front of this stage's parameter vectors
        ac.w[:n_ck], ac.theta_mu[:n_ck], ac.theta_sigma[:n_ck] = ck["w"], ck["theta_mu"], ck["theta_sigma"]

    if args.evaluate:
        n = args.eval_n
        r = ac.evaluate()

        def ci(p):                                   # 95% Wilson interval for a rate over n scenarios
            z = 1.96
            centre = (p + z * z / (2 * n)) / (1 + z * z / n)
            half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
            return f"{p:5.0%}  (95% CI {max(centre - half, 0):4.0%}-{min(centre + half, 1):4.0%})"
        print(f"stage {args.stage}, {n} scenarios, greedy policy from {src}")
        print("  goal reached      ", ci(r["success"]))
        print("  restricted entry  ", ci(r["restricted"]))
        print("  intrusion         ", ci(r["intrusion"]))
        print("  sector exit       ", ci(r["sector_exit"]))
        print(f"  mean flight time   {r['mean_time']:.0f} s (successful episodes)")
        env.close()
        return

    if args.watch:
        print("canonical:", ac.rollout("canonical"))
        for i in range(5):
            print(f"random {i}:", ac.rollout(ac.eval_scenarios[i]))
        env.close()
        return

    print(f"stage {args.stage}: {ac.features.size} features, {ac.features.n_active} active per state", flush=True)
    ac.train(args.episodes, eval_every=args.eval_every)
    np.savez(out, **ac.get_params())
    with open(out.replace(".npz", "_history.json"), "w") as f:
        json.dump(ac.history, f, indent=1, default=float)
    print("saved", out)
    env.close()


if __name__ == "__main__":
    main()
