"""Build policy_gradient/redblob_explainer.html (single self-contained page, works offline).

    python -m policy_gradient.build_redblob_explainer            # computes the data once, then builds
    python -m policy_gradient.build_redblob_explainer --recompute

Real scenarios come from the competition generator (core.scenario.ScenarioGenerator, single-agent defaults,
seed 42). For each one we compute, in a flat nautical-mile frame (no BlueSky):

  * the shortest path around the true polygons (visibility graph over polygon vertices) -- what a
    circular-obstacle planner would approach if it had the real shapes,
  * what a circular-obstacle planner gets from the enclosing circles the observation exposes
    (circles drawn as 24-gons; no path if the start or goal lies inside a circle),
  * what happens when the aircraft flies that static shortest path while the scripted intruders move
    (intruders follow their route at constant speed and keep their last heading afterwards: an assumption
    about BlueSky's behaviour that is not checked here).
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

from core.scenario import ScenarioGenerator          # noqa: E402
from policy_gradient.intruder_study import Route, to_nm, C0   # noqa: E402

N_GON = 24
SEP_NM = 5.0
V_EGO_NM_S = 150.0 / 1852.0
CAPTURE_NM = 5.0 / 1.852
N_STATS = 300
N_SHOW = 12
CACHE = os.path.join(HERE, "figures", "_redblob_data.json")


def shortest_path(start, goal, polys):
    """Dijkstra over the visibility graph of polygon vertices. Returns a list of (x, y) or None."""
    nodes = [tuple(start), tuple(goal)]
    for p in polys:
        nodes.extend(p.exterior.coords[:-1])
    shrunk = [p.buffer(-1e-6) for p in polys]
    segs = {}

    def visible(i, j):
        key = (i, j) if i < j else (j, i)
        if key not in segs:
            line = LineString([nodes[i], nodes[j]])
            segs[key] = not any(s.intersects(line) for s in shrunk if not s.is_empty)
        return segs[key]

    n = len(nodes)
    dist, prev = [math.inf] * n, [-1] * n
    dist[0] = 0.0
    pq = [(0.0, 0)]
    while pq:
        d, i = heapq.heappop(pq)
        if d > dist[i]:
            continue
        if i == 1:
            break
        for j in range(n):
            if j != i:
                w = math.dist(nodes[i], nodes[j])
                if d + w < dist[j] and visible(i, j):
                    dist[j], prev[j] = d + w, i
                    heapq.heappush(pq, (dist[j], j))
    if math.isinf(dist[1]):
        return None
    path, i = [], 1
    while i != -1:
        path.append(nodes[i])
        i = prev[i]
    return path[::-1]


def length(path):
    return sum(math.dist(a, b) for a, b in zip(path[:-1], path[1:]))


def corner_turns(path):
    out = []
    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        u, v = np.subtract(b, a), np.subtract(c, b)
        cosang = float(u @ v / (np.hypot(*u) * np.hypot(*v) + 1e-12))
        out.append(math.degrees(math.acos(max(-1, min(1, cosang)))))
    return out


def along(path, s):
    """Point at arc length s along a polyline (clamped to the end)."""
    for a, b in zip(path[:-1], path[1:]):
        d = math.dist(a, b)
        if s <= d:
            f = s / d if d else 0.0
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        s -= d
    return path[-1]


def flies_into_intruder(path, routes):
    """Fly the polyline at 150 m/s until 5 km short of the goal; True if any intruder comes within 5 NM."""
    total = max(length(path) - CAPTURE_NM, 0.0)
    ts = np.arange(0.0, total / V_EGO_NM_S + 1, 5.0)
    ego = np.array([along(path, V_EGO_NM_S * t) for t in ts])
    worst = math.inf
    for r in routes:
        ip = r.pos(ts)
        worst = min(worst, float(np.hypot(*(ego - ip).T).min()))
    return worst < SEP_NM, worst, float(ts[-1])


def main(recompute=False):
    if recompute or not os.path.exists(CACHE):
        compute()
    with open(CACHE) as f:
        data = json.load(f)
    with open(os.path.join(HERE, "redblob_explainer_template.html"), encoding="utf-8") as f:
        html = f.read()
    assert "__DATA__" in html
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = os.path.join(HERE, "redblob_explainer.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


def compute():
    with open(os.path.join(HERE, "figures", "_obstacle_geometry_study.json")) as f:
        geom = json.load(f)
    rng = np.random.default_rng(42)
    gen = ScenarioGenerator(n_obstacles=5, n_intruders=10, agent_speed=150.0, center=C0)
    shown, hit_poly, hit_straight, max_turns, ratios, n_ok = [], [], [], [], [], 0
    for i in range(N_STATS):
        sc = gen.generate(rng)
        ag = sc.agents[0]
        s, g = to_nm(ag.start), to_nm(ag.goal)
        polys = [Polygon([to_nm(v) for v in ob.vertices]) for ob in sc.obstacles]
        routes = [Route(r.waypoints, r.speed) for r in sc.intruder_routes]
        path = shortest_path(s, g, polys)
        if path is None:
            continue
        n_ok += 1
        straight = [s, g]
        hp, sep_p, t_p = flies_into_intruder(path, routes)
        hs, sep_s, t_s = flies_into_intruder(straight, routes)
        hit_poly.append(hp); hit_straight.append(hs)
        turns = corner_turns(path)
        max_turns.append(max(turns) if turns else 0.0)
        ratios.append(length(path) / math.dist(s, g))
        if len(shown) < N_SHOW:
            circs, cpolys = [], []
            for p in polys:
                pts = np.array(p.exterior.coords[:-1])
                c = pts.mean(axis=0)
                r = float(np.max(np.hypot(*(pts - c).T)))
                circs.append((c, r))
                cpolys.append(Polygon([(c[0] + r / math.cos(math.pi / N_GON) * math.cos(2 * math.pi * k / N_GON),
                                        c[1] + r / math.cos(math.pi / N_GON) * math.sin(2 * math.pi * k / N_GON))
                                       for k in range(N_GON)]))
            inside = [bool(any(cp.contains(Point(pt)) for cp in cpolys)) for pt in (s, g)]
            cpath = None if any(inside) else shortest_path(s, g, cpolys)
            flip = lambda p: [float(p[1]), float(p[0])]          # (north, east) -> plot (east, north)
            shown.append(dict(
                sector=[flip(to_nm(v)) for v in sc.sector],
                polys=[[flip(q) for q in p.exterior.coords[:-1]] for p in polys],
                circles=[dict(c=flip(c), r=r) for c, r in circs],
                start=flip(s), goal=flip(g),
                routes=[dict(pts=[flip(to_nm(w)) for w in r.waypoints], v=float(r.speed)) for r in sc.intruder_routes],
                straight=[flip(q) for q in straight], poly_path=[flip(q) for q in path],
                circle_path=[flip(q) for q in cpath] if cpath else None,
                start_in_circle=inside[0], goal_in_circle=inside[1],
                straight_len=math.dist(s, g), poly_len=length(path), circle_len=length(cpath) if cpath else None,
                corner_turns=[round(t, 1) for t in turns],
                poly_path_hits=bool(hp), poly_path_min_sep=float(sep_p), straight_hits=bool(hs), straight_min_sep=float(sep_s),
                t_poly=t_p, t_straight=t_s,
                area_ratio=[math.pi * r * r / p.area for (c, r), p in zip(circs, polys)]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{N_STATS}", flush=True)
    stats = dict(geom, n_paths=n_ok,
                 static_polygon_path_hits_intruder=float(np.mean(hit_poly)),
                 straight_line_hits_intruder=float(np.mean(hit_straight)),
                 median_max_corner_turn_deg=float(np.median(max_turns)),
                 share_corner_over_45=float(np.mean(np.array(max_turns) > 45)),
                 share_with_any_corner=float(np.mean(np.array(max_turns) > 1)),
                 median_poly_detour=float(np.median(ratios)))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(dict(stats=stats, scenarios=shown), f)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(recompute="--recompute" in sys.argv)
