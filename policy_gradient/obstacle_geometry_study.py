"""How circular is the competition's obstacle world, and how much does the straight line detour?

    python -m policy_gradient.obstacle_geometry_study [n_scenarios]

The competition obstacles are irregular polygons (3-8 vertices, stretched up to 2.5x), scored exactly as
polygons, but the observation only exposes a centroid and an enclosing circle. This script draws scenarios
from the single-agent generator and compares, per scenario:

  * how much bigger the enclosing circle is than the polygon (area ratio),
  * how often the start->goal straight line crosses a polygon vs an enclosing circle,
  * the shortest path around the polygons (true) vs around the enclosing circles (what a circular-obstacle
    planner would use), as a ratio to the straight line.

Shortest paths use a visibility graph (circles approximated by circumscribed 32-gons, so slightly
conservative). No BlueSky needed.
"""

import heapq
import json
import math
import os
import sys

import numpy as np
from shapely.geometry import LineString, Point, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from core.scenario import ScenarioGenerator   # noqa: E402

C0 = (52.0, 4.0)
CAPTURE_NM = 5.0 / 1.852
N_GON = 32


def to_nm(p):
    return (p[0] - C0[0]) * 60.0, (p[1] - C0[1]) * 60.0 * math.cos(math.radians(C0[0]))


def shortest_path_len(start, goal, obstacles):
    """Visibility-graph Dijkstra around a list of shapely polygons; start/goal tuples in NM."""
    nodes = [start, goal]
    for ob in obstacles:
        nodes.extend(ob.exterior.coords[:-1])
    shrunk = [ob.buffer(-1e-6) for ob in obstacles]

    def visible(a, b):
        seg = LineString([a, b])
        return not any(s.intersects(seg) for s in shrunk if not s.is_empty)

    n = len(nodes)
    dist = [math.inf] * n
    dist[0] = 0.0
    pq = [(0.0, 0)]
    while pq:
        d, i = heapq.heappop(pq)
        if d > dist[i]:
            continue
        if i == 1:
            return d
        for j in range(n):
            if j == i:
                continue
            w = math.dist(nodes[i], nodes[j])
            if d + w < dist[j] and visible(nodes[i], nodes[j]):
                dist[j] = d + w
                heapq.heappush(pq, (dist[j], j))
    return math.inf


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(42)
    gen = ScenarioGenerator(n_obstacles=5, n_intruders=10, agent_speed=150.0, center=C0)

    area_ratio, cross_poly, cross_circ = [], [], []
    detour_poly, detour_circ = [], []
    inside_circle, inside_poly = [], []
    n_obs_cross_poly, n_obs_cross_circ = [], []

    for i in range(n):
        sc = gen.generate(rng)
        ag = sc.agents[0]
        s, g = to_nm(ag.start), to_nm(ag.goal)
        straight = math.dist(s, g)
        polys = [Polygon([to_nm(v) for v in ob.vertices]) for ob in sc.obstacles]
        circs = []
        for p in polys:
            c = np.array(p.exterior.coords[:-1]).mean(axis=0)
            r = max(math.dist(c, v) for v in p.exterior.coords[:-1])
            circs.append((c, r))
            area_ratio.append(math.pi * r * r / p.area)
        circ_polys = [Polygon([(c[0] + r / math.cos(math.pi / N_GON) * math.cos(2 * math.pi * k / N_GON),
                                c[1] + r / math.cos(math.pi / N_GON) * math.sin(2 * math.pi * k / N_GON))
                               for k in range(N_GON)]) for c, r in circs]
        line = LineString([s, g])
        sp, gp = Point(s), Point(g)
        inside_circle.append(any(c.contains(sp) or c.contains(gp) for c in circ_polys))
        inside_poly.append(any(p.contains(sp) or p.contains(gp) for p in polys))
        hp = [p.intersects(line) for p in polys]
        hc = [p.intersects(line) for p in circ_polys]
        cross_poly.append(any(hp))
        cross_circ.append(any(hc))
        n_obs_cross_poly.append(sum(hp))
        n_obs_cross_circ.append(sum(hc))
        detour_poly.append(shortest_path_len(s, g, polys) / straight)
        detour_circ.append(shortest_path_len(s, g, circ_polys) / straight)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    def q(a):
        a = np.array(a)
        a = a[np.isfinite(a)]
        return dict(mean=float(a.mean()), median=float(np.median(a)), p90=float(np.percentile(a, 90)),
                    max=float(a.max()))

    out = dict(
        n_scenarios=n,
        circle_to_polygon_area_ratio=q(area_ratio),
        straight_line_hits_polygon=float(np.mean(cross_poly)),
        straight_line_hits_circle=float(np.mean(cross_circ)),
        mean_obstacles_on_line_polygon=float(np.mean(n_obs_cross_poly)),
        mean_obstacles_on_line_circle=float(np.mean(n_obs_cross_circ)),
        detour_ratio_polygon=q(detour_poly), detour_ratio_circle=q(detour_circ),
        detour_over_5pct_polygon=float(np.mean(np.array(detour_poly) > 1.05)),
        detour_over_5pct_circle=float(np.mean(np.array(detour_circ) > 1.05)),
        unreachable_polygon=float(np.mean(np.isinf(detour_poly))),
        unreachable_circle=float(np.mean(np.isinf(detour_circ))),
        start_or_goal_inside_circle=float(np.mean(inside_circle)),
        start_or_goal_inside_polygon=float(np.mean(inside_poly)),
    )
    path = os.path.join(HERE, "figures", "_obstacle_geometry_study.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
