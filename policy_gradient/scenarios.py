"""Small staged scenarios for developing policy-gradient agents on CompetitionEnv.

Stage 1  goal only                (no obstacles, no intruders)
Stage 2  goal + one obstacle      (obstacle sits near the straight line start -> goal)
Stage 3  goal + obstacle + one scripted intruder crossing the agent's path

Everything is built from ``core.scenario`` dataclasses, so the unmodified
``CompetitionEnv`` flies them (no change to the scored environment). The geometry is
done in a flat nautical-mile frame (x = north, y = east) around CENTER.

``StageGenerator.generate(rng)`` has the same interface as ``core.scenario.ScenarioGenerator``,
so it can be dropped into ``env.scenario_generator`` for seeded random training.
``canonical_scenario(stage)`` is one fixed layout, used as the "fixed start" evaluation.
"""

import math

import numpy as np

from core.scenario import (Scenario, Obstacle, Route, AgentSpec,
                           _nm_to_latlon, _bearing_nm)

CENTER = (52.0, 4.0)
NM2KM = 1.852
HALF_SECTOR = 60.0        # NM, square sector
AGENT_SPEED = 150.0       # m/s (matches CompetitionEnv's ac_spd)

STAGE_COUNTS = {1: (0, 0), 2: (1, 0), 3: (1, 1)}   # stage -> (n_obstacles, n_intruders)


def _ll(p):
    return _nm_to_latlon(CENTER, p)


def _square(cx, cy, half):
    return [(cx - half, cy - half), (cx - half, cy + half),
            (cx + half, cy + half), (cx + half, cy - half)]


def build_scenario(start, goal, heading=None, obstacles=(), intruders=(), half_sector=HALF_SECTOR):
    """Assemble a Scenario from NM-frame numbers.

    obstacles: iterable of (cx, cy, half_width)       -> square restricted areas
    intruders: iterable of (spawn_xy, exit_xy, speed) -> straight-line routes, speed in m/s
    heading:   initial heading in degrees (None: point at the goal)
    """
    h = half_sector
    sector = [_ll(p) for p in [(-h, -h), (h, -h), (h, h), (-h, h)]]
    obs = [Obstacle(vertices=[_ll(v) for v in _square(cx, cy, half)]) for cx, cy, half in obstacles]
    routes = [Route(ac_id=f"AC{i + 1}", waypoints=[_ll(a), _ll(b)], speed=spd)
              for i, (a, b, spd) in enumerate(intruders)]
    if heading is None:
        heading = _bearing_nm(start, goal)
    agent = AgentSpec(ac_id="KL001", start=_ll(start), goal=_ll(goal),
                      heading=heading, speed=AGENT_SPEED)
    return Scenario(center=CENTER, sector=sector, obstacles=obs, agents=[agent],
                    intruder_routes=routes)


def canonical_scenario(stage, scale=1.0):
    """One fixed layout per stage: fly south-to-north across the middle of the sector.
    ``scale`` shrinks all the geometry (1.0 = the size of the competition sectors; speeds are
    unchanged, so smaller scenarios are simply shorter episodes)."""
    k = scale
    start, goal = (-40.0 * k, 0.0), (40.0 * k, 0.0)
    obstacles = [(0.0, 0.0, 10.0 * k)] if stage >= 2 else []
    intruders = []
    if stage >= 3:
        # crosses the agent's line a bit past the obstacle, from east to west, timed to meet it
        intruders = [((22.0 * k, 62.0 * k), (22.0 * k, -45.0 * k), 150.0)]
    return build_scenario(start, goal, obstacles=obstacles, intruders=intruders,
                          half_sector=HALF_SECTOR * k)


class StageGenerator:
    """Seeded random scenarios for one stage. generate(rng) takes a numpy Generator.

    With probability ``fixed_prob`` it returns the canonical layout instead, so a fraction of
    training episodes always starts from the same place (like the fixed start in the Dubins runs).
    """

    def __init__(self, stage, fixed_prob=0.25, scale=1.0, margin=10.0):
        assert stage in STAGE_COUNTS
        self.stage = stage
        self.fixed_prob = fixed_prob
        self.scale = scale
        self.limit = (HALF_SECTOR - margin) * scale   # start/goal stay this far inside the sector edge

    def generate(self, rng):
        if rng.random() < self.fixed_prob:
            return canonical_scenario(self.stage, self.scale)
        return self._random(rng)

    def _random(self, rng):
        for _ in range(1000):
            start = rng.uniform(-self.limit, self.limit, 2)
            k = self.scale
            dist_nm = rng.uniform(80, 150) * k / NM2KM
            ang = rng.uniform(0, 2 * math.pi)
            goal = start + dist_nm * np.array([math.cos(ang), math.sin(ang)])
            if np.abs(goal).max() > self.limit:
                continue
            obstacles, intruders = [], []
            if self.stage >= 2:
                half = rng.uniform(7, 12) * k
                t = rng.uniform(0.35, 0.65)
                along = start + t * (goal - start)
                unit = (goal - start) / dist_nm
                perp = np.array([-unit[1], unit[0]])
                centre = along + perp * rng.uniform(-0.6, 0.6) * half
                # the obstacle must not swallow (or crowd) the start or the goal
                if (np.hypot(*(centre - start)) < half * 1.5 + 10 * k
                        or np.hypot(*(centre - goal)) < half * 1.5 + 10 * k):
                    continue
                obstacles = [(centre[0], centre[1], half)]
            if self.stage >= 3:
                intruders = [self._intruder(rng, start, goal, dist_nm, k)]
            heading = math.degrees(rng.uniform(-math.pi, math.pi)) % 360.0   # random initial heading
            return build_scenario(tuple(start), tuple(goal), heading=heading,
                                  obstacles=obstacles, intruders=intruders,
                                  half_sector=HALF_SECTOR * k)
        raise RuntimeError("StageGenerator: could not place a scenario")

    @staticmethod
    def _intruder(rng, start, goal, dist_nm, k):
        """An intruder that crosses the agent's start->goal line roughly when the agent gets there."""
        speed = rng.uniform(120, 180)
        t = rng.uniform(0.45, 0.8)
        cross = start + t * (goal - start)
        unit = (goal - start) / dist_nm
        # crossing direction: 60..120 degrees off the agent's track, either side
        off = math.radians(rng.uniform(60, 120)) * rng.choice([-1, 1])
        d = np.array([unit[0] * math.cos(off) - unit[1] * math.sin(off),
                      unit[0] * math.sin(off) + unit[1] * math.cos(off)])
        t_agent_s = t * dist_nm * NM2KM * 1000 / AGENT_SPEED          # when the agent reaches `cross`
        t_arrive = max(t_agent_s + rng.uniform(-120, 120), 60.0)       # when the intruder does
        lead_nm = speed * t_arrive / 1000 / NM2KM                      # how far it flies before `cross`
        spawn = cross - d * lead_nm
        exit_ = cross + d * 60.0 * k
        return (tuple(spawn), tuple(exit_), speed)
