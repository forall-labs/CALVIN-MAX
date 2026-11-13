"""Calibrate camera_shift magnitudes by image-space displacement (D-026). Pure geometry, no simulator.

The static camera (calvin_env conf/cameras/cameras/static.yaml) is ~4 m away with a 10 deg FOV. We orbit
it about its look-at point and report the mean pixel displacement of workspace points (table surface,
slider shelves, drawer front) in the 200x200 image. We then pick orbit angles for target displacements
(A: 5% of image width, B: 10% plus a 1 deg FOV change).

    python scripts/calibrate_camera.py
"""

from __future__ import annotations

import json

import numpy as np

CAM = {"fov": 10.0, "width": 200, "height": 200,
       "look_at": [-0.026242351159453392, -0.0302329882979393, 0.3920000493526459],
       "look_from": [2.871459009488717, -2.166602199425597, 2.555159848480571],
       "up": [0.4041403970338857, 0.22629790978217404, 0.8862616969685161]}
# Workspace points: table surface corners, slider shelves, drawer front (scene D), z ~ 0.46.
WS = np.array([[0.0, -0.15, 0.46], [0.35, -0.15, 0.46], [0.0, -0.03, 0.46], [0.35, -0.03, 0.46],
               [-0.32, 0.05, 0.46], [-0.16, 0.12, 0.46], [-0.05, 0.05, 0.46], [0.13, 0.12, 0.46],
               [0.1, -0.2, 0.35], [0.25, -0.2, 0.35]])


def look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return s, u, f


def project(points, eye, target, up, fov_deg, w, h):
    s, u, f = look_at(np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float))
    rel = points - eye
    x, y, z = rel @ s, rel @ u, rel @ f
    t = np.tan(np.deg2rad(fov_deg) / 2)
    px = (x / (z * t) * 0.5 + 0.5) * w
    py = (0.5 - y / (z * t) * 0.5) * h
    return np.stack([px, py], axis=1)


def displacement(payload):
    from calvin_max.events import shifted_static_view

    eye, tgt, up = (np.array(CAM[k], float) for k in ("look_from", "look_at", "up"))
    p0 = project(WS, eye, tgt, up, CAM["fov"], CAM["width"], CAM["height"])
    e1, t1 = shifted_static_view(eye, tgt, up, payload)
    p1 = project(WS, e1, t1, up, CAM["fov"] + payload.get("fov_delta_deg", 0.0), CAM["width"], CAM["height"])
    return float(np.mean(np.linalg.norm(p1 - p0, axis=1)))


def main():
    unit = {"1cm_x": displacement({"translate_m": [0.01, 0, 0]}), "1cm_z": displacement({"translate_m": [0, 0, 0.01]}),
            "yaw_1deg": displacement({"yaw_deg": 1}), "pitch_1deg": displacement({"pitch_deg": 1}),
            "fov_+1deg": displacement({"fov_delta_deg": 1})}
    w = CAM["width"]
    # Shape of each variant is fixed (translate along x, yaw, pitch in a fixed ratio); scale s is solved
    # so the mean workspace displacement hits the target. B also widens the FOV by 1 deg.
    # Rotation-led shapes: translation and in-place rotation partly cancel in the image if mixed carelessly.
    shapes = {"A": ({"translate_m": [0.0, 0, 0], "yaw_deg": 1.0, "pitch_deg": 0.5}, 0.05 * w, 0.0),
              "B": ({"translate_m": [0.0, 0, 0.01], "yaw_deg": 1.0, "pitch_deg": 0.5}, 0.10 * w, 1.0)}
    chosen = {}
    for name, (shape, target, dfov) in shapes.items():
        def at(sc):
            return {"translate_m": [x * sc for x in shape["translate_m"]], "yaw_deg": shape["yaw_deg"] * sc,
                    "pitch_deg": shape["pitch_deg"] * sc, "fov_delta_deg": dfov}
        sc = min(np.arange(0.05, 5.0, 0.01), key=lambda v: abs(displacement(at(v)) - target))
        p = at(float(sc))
        chosen[name] = {**{k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for k, v in p.items()},
                        "mean_px_displacement": round(displacement(p), 2), "target_px": target}
    print(json.dumps({"px_per_unit": {k: round(v, 2) for k, v in unit.items()}, "chosen": chosen}, indent=1))


if __name__ == "__main__":
    main()
