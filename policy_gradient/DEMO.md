# 1:1 demo: policy gradient on the competition environment

## Run it (from the repo root, in PowerShell)

```powershell
.\.venv\Scripts\python.exe -m policy_gradient.demo
```

Opens two windows: a one-page strategy slide, and an animated side-by-side replay (baseline vs learned)
of scenario 28 with the live numbers. Both runs are flown on the real `CompetitionEnv` first (about 20 s),
then replayed. If a window does not open, use the saved files in `policy_gradient/figures/`:
`demo_scenario28.gif` (the replay), `demo_slide.png` (the slide), `demo_frame.png` (one still).

```powershell
.\.venv\Scripts\python.exe -m policy_gradient.demo --scenario 64      # another scenario (0-99)
.\.venv\Scripts\python.exe -m policy_gradient.demo --baseline policy_gradient/checkpoints/stage3_cpa.npz   # vs heading-only
.\.venv\Scripts\python.exe -m policy_gradient.demo --find             # list scenarios where pursuit fails and we do not
```

Companion pages (open in a browser): `obs_explainer.html` (the 131 observation numbers, interactive),
`cso_explainer.html` (contextual stochastic optimization), `explainer.html` (the method and how it differs from the Dubins case).

## Five-minute story

1. **What we did.** The environment and scoring are untouched. We change the three hooks and add a learner
   outside them: a one-step actor-critic (Sutton & Barto ch. 13) with a Gaussian policy over tile-coded features.
2. **What changed in each hook.** Observation: 131 numbers become about six physical quantities (goal, obstacle
   clearance, nearest intruder, straight-line closest approach, own speed). Action: heading plus, new this week,
   speed. Reward: minus one per second flown, goal progress, penalties read from the scored metrics.
3. **What you see in the replay.** Left: a hand-coded go-to-goal pursuit that ignores traffic. Right: the learned
   policy. The purple segment is the straight-line closest-approach prediction our features are built from;
   the readout under each map is exactly what the policy sees and sends.
4. **Result (100 fixed toy scenarios).**

   | | goal | restricted entry | intrusion | exit | mean time |
   |---|---|---|---|---|---|
   | Go-to-goal pursuit (hand-coded) | 100% | 100% | 66% | 0% | 401 s |
   | Heading only (learned) | 100% | 2% | 7% | 0% | 481 s |
   | Heading + speed (learned) | 100% | 8% | 5% | 0% | 447 s |

5. **Honest limits.** Half-scale toy scenarios (one square obstacle, one straight-line intruder; the competition
   has 5 obstacles and 10 intruders), one training seed, 100 scenarios per row so the error bars are wide. These
   are not competition scores.

## What we learned this week (all measured)

- **Speed.** BlueSky allows 67-163 m/s calibrated airspeed (165.6 m/s true at the top), so speeding up is worth
  at most about 9% of flight time; slowing is the bigger lever. The learned speed policy cut mean flight time from
  476 s (heading-only trained the same number of extra episodes) to 447 s, but it speeds up *everywhere*, including
  when a conflict is imminent. It is not yet timing crossings. With one intruder, urgent threats are only 1.6% of steps.
- **Continued training can hurt.** Training the heading-only policy 4000 more episodes raised restricted-area entries
  from 2% to 10% (overlapping error bars, but worth watching).
- **Intruders are mostly predictable.** On 1000 generated competition scenarios, half the intruder routes bend (median
  14 degrees, about 470 s in), yet a straight-line closest-approach check flags real 5-minute conflicts with 98%
  precision and 97.6% recall. Prediction headroom in the single-agent track is small.
- **Obstacles are polygons, not circles.** The observation only gives a centroid and an enclosing circle; in 65% of
  generated scenarios the start or goal lies inside that circle. A circle-based path planner would be wrong most of
  the time, and it is close to the "no classical planner" rule, so it is parked.

## Questions worth putting to the professor

- By "scalable method for dynamic obstacles", does she mean compute per decision as the number of intruders grows,
  sample efficiency, or model size? (Expected cost adds over obstacles, so a closest-approach risk profile scales linearly.)
- Is a hybrid acceptable (a risk model feeding the RL policy's features or reward)? The competition rules require an
  RL-based solution, so this is worth confirming with the organizers.
- The multi-agent track breaks the "intruders ignore us" assumption that contextual stochastic optimization relies on.
  Is that where she wants the novelty?

## Files

`policy_gradient/demo.py` (this demo), `nav_env.py` (hooks and reward), `actor_critic.py` (learner and features),
`train.py` (training and evaluation), `intruder_study.py`, `obstacle_geometry_study.py` (the two data studies).
