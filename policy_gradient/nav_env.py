"""A small, Dubins-style interface on top of the (unmodified) CompetitionEnv.

    state            = reset(scenario=None)
    state, reward, terminal, truncated, info = step(a)      a in [-1, 1]: turn right (+) / left (-)

What changes compared with ``gym.make("CompetitionEnv-v0")``:

* one continuous action, the heading change (speed is held), so a Gaussian policy fits; with
  ``speed=True`` there is a second one, the speed change (+ = accelerate, a fraction of ``d_speed`` per step);
* the 131-number observation is boiled down to a few physical quantities (``State``);
* our own reward (like the Dubins one, in units of ``reward_scale`` seconds):

      -dt                                  every second of flying costs 1
      + (d(s) - d(s')) / v                 potential-based shaping: progress towards the goal
      + goal_bonus                         for reaching the goal
      - lam * (seconds in a restricted area / outside the sector / in loss of separation)
      - event_penalty * (entries into restricted area / sector exits / intrusions)

  The penalties are read from the env's scored metrics (``info``), as deltas over the step.
  Unlike the Dubins car nothing is terminal except the goal, so entering an obstacle is just costly.

Angles: every relative bearing is "target minus own heading", in radians in (-pi, pi]:
positive = target is to the right (starboard). An action > 0 turns right.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

import bluesky_gym  # noqa: F401  (registers envs)
from bluesky_gym.envs.competition_env import CompetitionEnv

from policy_gradient.scenarios import StageGenerator, STAGE_COUNTS, canonical_scenario, AGENT_SPEED

NM2KM = 1.852
OBSTACLE_RADIUS_NORM_NM = math.sqrt(1000.0 / math.pi)   # CompetitionEnv's radius_norm


@dataclass
class State:
    dist_goal_km: float
    goal_bearing: float                        # rad
    obstacle: Optional[tuple] = None           # (clearance_nm to the nearest obstacle's edge, bearing rad)
    # (distance_nm, bearing rad, aspect rad, time to closest approach s, signed miss distance nm;
    #  miss distance > 0: the intruder will pass on our right)
    intruder: Optional[tuple] = None
    speed_ms: float = AGENT_SPEED              # own true airspeed (m/s); only varies when NavEnv(speed=True)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class NavEnv:
    def __init__(self, stage=1, fixed_prob=0.25, scale=0.5, action_frequency=10, seed=0,
                 goal_bonus=5.0, lam=0.01, event_penalty=0.5, reward_scale=100.0,
                 render_mode=None, speed=False):
        n_obs, n_int = STAGE_COUNTS[stage]
        self.stage, self.scale = stage, scale
        self.goal_bonus, self.lam = goal_bonus, lam
        self.event_penalty, self.reward_scale = event_penalty, reward_scale
        self.speed = speed
        self.env = CompetitionEnv(render_mode=render_mode, n_obstacles=n_obs, n_intruders=n_int,
                                  action_frequency=action_frequency)
        self.env.scenario_generator = StageGenerator(stage, fixed_prob, scale)   # same .generate(rng) interface
        self.max_step_seconds = action_frequency * self.env.sim_dt
        self._seed = seed
        self._first_reset = True

    # ------------------------------------------------------------------ API
    def reset(self, scenario=None):
        """scenario: None = random from the StageGenerator, "canonical" = the fixed layout,
        or a core.scenario.Scenario."""
        if scenario == "canonical":
            scenario = canonical_scenario(self.stage, self.scale)
        self.env._fixed_scenario = scenario
        seed, self._seed = self._seed, None      # seed the env's RNG once, on the very first reset
        obs, info = self.env.reset(seed=seed)
        self.info = info
        self.state = self._state(obs)
        return self.state

    def step(self, a):
        """a: heading action (float), or (heading, speed) when speed=True; each in [-1, 1]."""
        a_h, a_s = (a if self.speed else (a, 0.0))
        action = np.clip(np.array([a_h, a_s], dtype=float), -1.0, 1.0)
        before, d0 = self.info, self.state.dist_goal_km
        obs, _, terminal, truncated, info = self.env.step(action)
        self.info = info
        self.state = self._state(obs)

        dt = info["flight_time"] - before["flight_time"]
        d1 = 0.0 if terminal else self.state.dist_goal_km         # potential is 0 at the goal
        v_kms = AGENT_SPEED / 1000.0
        reward = -dt + (d0 - d1) / v_kms                           # seconds
        if terminal:
            reward += self.goal_bonus * self.reward_scale
        for t_key, e_key in (("time_in_restricted_area", "restricted_area_events"),
                             ("time_outside_sector", "sector_exit_events"),
                             ("intrusion_time", "intrusion_events")):
            reward -= self.lam * self.reward_scale * (info[t_key] - before[t_key])
            reward -= self.event_penalty * self.reward_scale * (info[e_key] - before[e_key])
        return self.state, reward / self.reward_scale, terminal, truncated, info

    def close(self):
        self.env.close()

    # ---------------------------------------------------------------- state
    def _state(self, obs):
        goal_bearing = wrap(-math.atan2(obs["sin_drift"][0], obs["cos_drift"][0]))   # drift = hdg - qdr
        s = State(dist_goal_km=float(obs["waypoint_distance"][0]) * self.env.waypoint_obs.distance_norm,
                  goal_bearing=goal_bearing,
                  speed_ms=float(obs["airspeed"][0]) * self.env.airspeed_obs.spd_std + self.env.airspeed_obs.spd_mean)
        if len(obs["obstacle_distance"]):
            o = self.env.obstacle_obs
            centre_nm = float(obs["obstacle_distance"][0]) * o.distance_norm / NM2KM
            radius_nm = float(obs["obstacle_radius"][0]) * o.radius_norm
            bearing = wrap(-math.atan2(obs["obstacle_sin_bearing"][0], obs["obstacle_cos_bearing"][0]))
            s.obstacle = (centre_nm - radius_nm, bearing)
        if len(obs["x_r"]):
            x, y = float(obs["x_r"][0]), float(obs["y_r"][0])                       # body frame, +y = right
            bearing = math.atan2(y, x)
            track = math.atan2(float(obs["vy_r"][0]), float(obs["vx_r"][0]))        # direction of relative velocity
            aspect = wrap(track - (bearing + math.pi))   # 0: the intruder is flying straight at us
            # closest point of approach, assuming both keep their current velocity (body frame, metres, m/s)
            io = self.env.intruder_obs
            rx, ry = x * io.pos_norm, y * io.pos_norm
            vx, vy = float(obs["vx_r"][0]) * io.spd_norm, float(obs["vy_r"][0]) * io.spd_norm
            v2 = vx * vx + vy * vy
            tcpa = max(-(rx * vx + ry * vy) / v2, 0.0) if v2 > 1e-9 else 0.0     # s; 0 if diverging
            cx, cy = rx + vx * tcpa, ry + vy * tcpa
            dcpa_nm = math.copysign(math.hypot(cx, cy) / 1000.0 / NM2KM, cy)    # >0: passes on our right
            s.intruder = (float(obs["intruder_distance"][0]) * io.dist_norm, bearing, aspect, tcpa, dcpa_nm)
        return s
