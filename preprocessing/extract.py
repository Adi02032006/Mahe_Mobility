#WILL BE USING THIS TO EXTRACT TRAJECTORIES FROM THE NUSCENES JSON
"""
preprocessing/extract.py
Walks nuScenes linked-list annotations → clean per-agent trajectories.
"""
import json
from pathlib import Path

PEDESTRIAN_CATS = {
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.wheelchair",
    "human.pedestrian.stroller",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.construction_worker",
}
CYCLIST_CATS = {"vehicle.bicycle", "vehicle.motorcycle"}
TARGET_CATS  = PEDESTRIAN_CATS | CYCLIST_CATS


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_trajectories(data_root: str, min_track_len: int = 10):
    """
    Returns dict: instance_token -> list of dicts
        {x, y, timestamp, visibility, category, is_cyclist}
    sorted by timestamp.
    """
    root = Path(data_root)
    categories  = load_json(root / "category.json")
    instances   = load_json(root / "instance.json")
    annotations = load_json(root / "sample_annotation.json")
    samples     = load_json(root / "sample.json")
    vis_map     = load_json(root / "visibility.json")

    cat_name    = {c["token"]: c["name"] for c in categories}
    inst_cat    = {i["token"]: cat_name.get(i["category_token"], "")
                   for i in instances}
    sample_ts   = {s["token"]: s["timestamp"] for s in samples}
    vis_level   = {v["token"]: int(v["token"]) for v in vis_map}
    ann_by_tok  = {a["token"]: a for a in annotations}

    trajectories = {}
    for ann in annotations:
        if ann["prev"] != "":
            continue  # start from head only
        inst  = ann["instance_token"]
        cat   = inst_cat.get(inst, "")
        if cat not in TARGET_CATS:
            continue

        traj, cur = [], ann
        while cur:
            x, y, _ = cur["translation"]
            traj.append({
                "x":          float(x),
                "y":          float(y),
                "timestamp":  sample_ts.get(cur["sample_token"], 0),
                "visibility": vis_level.get(cur.get("visibility_token", "4"), 4),
                "category":   cat,
                "is_cyclist": cat in CYCLIST_CATS,
            })
            nxt = cur["next"]
            cur = ann_by_tok.get(nxt) if nxt else None

        if len(traj) >= min_track_len:
            trajectories[inst] = sorted(traj, key=lambda d: d["timestamp"])

    print(f"[extract] {len(trajectories)} valid tracks (min_len={min_track_len})")
    return trajectories


if __name__ == "__main__":
    import sys
    trajs = extract_trajectories(sys.argv[1] if len(sys.argv) > 1 else "data/v1.0-mini")
    for tok, t in list(trajs.items())[:3]:
        print(f"  {tok[:8]}… len={len(t)} cat={t[0]['category']}")
