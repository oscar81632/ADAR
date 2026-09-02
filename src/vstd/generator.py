import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml
from PIL import Image, ImageDraw


Cell = Tuple[int, int]


DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass(frozen=True)
class VSTDClass:
    id: int
    name: str
    prompt: str
    color: str
    shape: str
    rgb: Tuple[int, int, int]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_classes(config: dict) -> List[VSTDClass]:
    colors = config["classes"]["colors"]
    shapes = config["classes"]["shapes"]
    classes = []
    for color_name, rgb in colors.items():
        for shape in shapes:
            name = f"{color_name} {shape}"
            classes.append(
                VSTDClass(
                    id=len(classes),
                    name=name,
                    prompt=f"a path ending at a {name}",
                    color=color_name,
                    shape=shape,
                    rgb=tuple(int(x) for x in rgb),
                )
            )
    return classes


def in_bounds(cell: Cell, grid_size: int) -> bool:
    r, c = cell
    return 0 <= r < grid_size and 0 <= c < grid_size


def direction_between(a: Cell, b: Cell) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    for name, delta in DIRECTIONS.items():
        if delta == (dr, dc):
            return name
    raise ValueError(f"Cells are not adjacent: {a} -> {b}")


def count_turns(path: Sequence[Cell]) -> int:
    if len(path) < 3:
        return 0
    directions = [direction_between(path[i], path[i + 1]) for i in range(len(path) - 1)]
    return sum(1 for i in range(1, len(directions)) if directions[i] != directions[i - 1])


def path_bbox_area(path: Sequence[Cell]) -> int:
    rows = [cell[0] for cell in path]
    cols = [cell[1] for cell in path]
    return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def path_metrics(path: Sequence[Cell], grid_size: int) -> dict:
    start = path[0]
    end = path[-1]
    path_length = len(path) - 1
    manhattan = abs(start[0] - end[0]) + abs(start[1] - end[1])
    bbox_area = path_bbox_area(path)
    coverage = len(set(path)) / float(grid_size * grid_size)
    normalized_length = path_length / max(1, grid_size * grid_size - 1)
    normalized_distance = manhattan / max(1, 2 * (grid_size - 1))
    normalized_bbox = bbox_area / float(grid_size * grid_size)
    long_dependency_score = (
        0.50 * normalized_length
        + 0.30 * normalized_distance
        + 0.20 * normalized_bbox
    )
    return {
        "start_end_manhattan": manhattan,
        "num_turns": count_turns(path),
        "path_bbox_area": bbox_area,
        "path_coverage": coverage,
        "long_dependency_score": long_dependency_score,
    }


def random_path(
    rng: random.Random,
    grid_size: int,
    path_length: int,
    min_turns: int,
    max_attempts: int,
    start: Cell | None = None,
    min_manhattan: int = 0,
) -> List[Cell]:
    for _ in range(max_attempts):
        path_start = start if start is not None else (rng.randrange(grid_size), rng.randrange(grid_size))
        path = [path_start]
        visited = {path_start}
        turns = 0
        prev_dir = None

        for _step in range(path_length):
            current = path[-1]
            candidates = []
            for direction, (dr, dc) in DIRECTIONS.items():
                nxt = (current[0] + dr, current[1] + dc)
                if in_bounds(nxt, grid_size) and nxt not in visited:
                    candidates.append((direction, nxt))
            if not candidates:
                break
            direction, nxt = rng.choice(candidates)
            if prev_dir is not None and direction != prev_dir:
                turns += 1
            prev_dir = direction
            path.append(nxt)
            visited.add(nxt)

        manhattan = abs(path[-1][0] - path[0][0]) + abs(path[-1][1] - path[0][1])
        if len(path) == path_length + 1 and turns >= min_turns and manhattan >= min_manhattan:
            return path

    raise RuntimeError(f"Unable to generate path length={path_length} in {max_attempts} attempts")


def random_path_avoiding(
    rng: random.Random,
    grid_size: int,
    path_length: int,
    min_turns: int,
    avoid: Sequence[Cell],
    max_attempts: int,
    start_pool: Sequence[Cell] | None = None,
) -> List[Cell]:
    avoid_set = set(avoid)
    for _ in range(max_attempts):
        if start_pool:
            start = rng.choice(list(start_pool))
        else:
            start = (rng.randrange(grid_size), rng.randrange(grid_size))
        if start in avoid_set:
            continue
        path = [start]
        visited = {start}
        turns = 0
        prev_dir = None

        for _step in range(path_length):
            current = path[-1]
            candidates = []
            for direction, (dr, dc) in DIRECTIONS.items():
                nxt = (current[0] + dr, current[1] + dc)
                if in_bounds(nxt, grid_size) and nxt not in visited and nxt not in avoid_set:
                    candidates.append((direction, nxt))
            if not candidates:
                break
            direction, nxt = rng.choice(candidates)
            if prev_dir is not None and direction != prev_dir:
                turns += 1
            prev_dir = direction
            path.append(nxt)
            visited.add(nxt)

        if len(path) == path_length + 1 and turns >= min_turns:
            return path

    raise RuntimeError(f"Unable to generate decoy path length={path_length} in {max_attempts} attempts")


def cell_center(cell: Cell, image_size: int, grid_size: int) -> Tuple[float, float]:
    cell_size = image_size / grid_size
    r, c = cell
    return ((c + 0.5) * cell_size, (r + 0.5) * cell_size)


def cell_box(cell: Cell, image_size: int, grid_size: int, scale: float = 0.58) -> Tuple[float, float, float, float]:
    cx, cy = cell_center(cell, image_size, grid_size)
    half = image_size / grid_size * scale / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def draw_grid(draw: ImageDraw.ImageDraw, image_size: int, grid_size: int):
    cell_size = image_size / grid_size
    grid_color = (226, 226, 222)
    for i in range(grid_size + 1):
        x = round(i * cell_size)
        y = round(i * cell_size)
        draw.line([(x, 0), (x, image_size)], fill=grid_color, width=1)
        draw.line([(0, y), (image_size, y)], fill=grid_color, width=1)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    cell: Cell,
    direction: str,
    image_size: int,
    grid_size: int,
    color,
    width: int,
    length_scale: float = 0.55,
    head_scale: float = 0.18,
):
    cx, cy = cell_center(cell, image_size, grid_size)
    cell_size = image_size / grid_size
    length = cell_size * length_scale
    head = cell_size * head_scale
    dr, dc = DIRECTIONS[direction]
    end = (cx + dc * length / 2, cy + dr * length / 2)
    start = (cx - dc * length / 2, cy - dr * length / 2)
    draw.line([start, end], fill=color, width=width)

    angle = math.atan2(dr, dc)
    left = (
        end[0] - head * math.cos(angle - math.pi / 6),
        end[1] - head * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head * math.cos(angle + math.pi / 6),
        end[1] - head * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, left, right], fill=color)


def star_points(cx: float, cy: float, outer: float, inner: float, points: int = 5):
    coords = []
    for i in range(points * 2):
        radius = outer if i % 2 == 0 else inner
        angle = -math.pi / 2 + i * math.pi / points
        coords.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return coords


def draw_start_marker(draw: ImageDraw.ImageDraw, cell: Cell, image_size: int, grid_size: int):
    cell_size = image_size / grid_size
    r, c = cell
    pad = max(1, int(cell_size * 0.08))
    width = max(3, int(cell_size * 0.16))
    x0 = c * cell_size + pad
    y0 = r * cell_size + pad
    x1 = (c + 1) * cell_size - pad
    y1 = (r + 1) * cell_size - pad
    draw.rectangle((x0, y0, x1, y1), outline=(112, 66, 190), width=width)


def draw_shape(draw: ImageDraw.ImageDraw, cell: Cell, shape: str, rgb, image_size: int, grid_size: int):
    box = cell_box(cell, image_size, grid_size, scale=0.62)
    outline = (40, 40, 40)
    if shape == "circle":
        draw.ellipse(box, fill=rgb, outline=outline, width=2)
    elif shape == "square":
        draw.rectangle(box, fill=rgb, outline=outline, width=2)
    elif shape == "triangle":
        x0, y0, x1, y1 = box
        draw.polygon([(x0 + x1) / 2, y0, x0, y1, x1, y1], fill=rgb, outline=outline)
    else:
        raise ValueError(f"Unsupported shape: {shape}")


def sample_distractors(
    rng: random.Random,
    grid_size: int,
    true_path: Sequence[Cell],
    ratio: float,
    classes: Sequence[VSTDClass],
    allow_shapes: bool,
    black_arrow_prob: float = 0.0,
    extra_occupied: Sequence[Cell] = (),
) -> List[dict]:
    occupied = set(true_path)
    occupied.update(extra_occupied)
    available = [(r, c) for r in range(grid_size) for c in range(grid_size) if (r, c) not in occupied]
    rng.shuffle(available)
    count = min(len(available), int(round(len(available) * ratio)))
    distractors = []
    for cell in available[:count]:
        if allow_shapes and rng.random() < 0.25:
            cls = rng.choice(classes)
            distractors.append(
                {
                    "cell": list(cell),
                    "type": "shape",
                    "class_name": cls.name,
                    "color": cls.color,
                    "shape": cls.shape,
                }
            )
        else:
            item = {
                "cell": list(cell),
                "type": "arrow",
                "direction": rng.choice(list(DIRECTIONS.keys())),
            }
            if rng.random() < black_arrow_prob:
                item["color"] = "black"
            distractors.append(item)
    return distractors


def sample_decoy_paths(
    rng: random.Random,
    config: dict,
    generation: dict,
    grid_size: int,
    true_path: Sequence[Cell],
    classes: Sequence[VSTDClass],
    existing_distractors: Sequence[dict],
) -> List[dict]:
    decoy_cfg = generation.get("decoy_paths", {})
    count = int(decoy_cfg.get("count", 0))
    if count <= 0:
        return []

    min_length = int(decoy_cfg.get("min_length", 6))
    max_length = int(decoy_cfg.get("max_length", min_length))
    min_turns = int(decoy_cfg.get("min_turns", 1))
    max_attempts = int(generation.get("max_attempts_per_sample", 200))
    arrow_color = str(decoy_cfg.get("arrow_color", "black"))
    endpoint_shape_prob = float(decoy_cfg.get("endpoint_shape_prob", 1.0))
    start_near_true_start_radius = decoy_cfg.get("start_near_true_start_radius")
    start_pool = None
    if start_near_true_start_radius is not None:
        radius = int(start_near_true_start_radius)
        true_start = true_path[0]
        start_pool = [
            (r, c)
            for r in range(grid_size)
            for c in range(grid_size)
            if abs(r - true_start[0]) + abs(c - true_start[1]) <= radius
        ]

    occupied = set(true_path)
    occupied.update(tuple(item["cell"]) for item in existing_distractors)
    items = []
    for _ in range(count):
        length = rng.randint(min_length, max_length)
        path = random_path_avoiding(
            rng,
            grid_size=grid_size,
            path_length=length,
            min_turns=min_turns,
            avoid=occupied,
            max_attempts=max_attempts,
            start_pool=start_pool,
        )
        occupied.update(path)
        for i in range(len(path) - 1):
            items.append(
                {
                    "cell": list(path[i]),
                    "type": "arrow",
                    "direction": direction_between(path[i], path[i + 1]),
                    "color": arrow_color,
                    "decoy_path": True,
                }
            )
        if rng.random() < endpoint_shape_prob:
            cls = rng.choice(classes)
            items.append(
                {
                    "cell": list(path[-1]),
                    "type": "shape",
                    "class_name": cls.name,
                    "color": cls.color,
                    "shape": cls.shape,
                    "decoy_path": True,
                }
            )
    return items


def render_sample(config: dict, classes: Sequence[VSTDClass], sample: dict) -> Image.Image:
    image_cfg = config["image"]
    image_size = int(image_cfg["size"])
    grid_size = int(image_cfg["grid_size"])
    background = tuple(int(x) for x in image_cfg["background"])
    line_width = int(image_cfg.get("line_width", 3))

    image = Image.new("RGB", (image_size, image_size), background)
    draw = ImageDraw.Draw(image)
    draw_grid(draw, image_size, grid_size)

    if config["generation"].get("draw_path_trace", False):
        centers = [cell_center(tuple(cell), image_size, grid_size) for cell in sample["path"]]
        draw.line(centers, fill=(198, 198, 198), width=max(1, line_width - 1))

    for item in sample["distractors"]:
        cell = tuple(item["cell"])
        if item["type"] == "arrow":
            arrow_color = (8, 8, 8) if item.get("color") == "black" else (160, 160, 160)
            draw_arrow(
                draw,
                cell,
                item["direction"],
                image_size,
                grid_size,
                color=arrow_color,
                width=line_width,
                length_scale=0.62,
                head_scale=0.26,
            )
        else:
            cls = next(c for c in classes if c.name == item["class_name"])
            rgb = cls.rgb if item.get("decoy_path") else tuple(max(0, x - 45) for x in cls.rgb)
            draw_shape(draw, cell, cls.shape, rgb, image_size, grid_size)

    for arrow in sample["arrows"]:
        draw_arrow(
            draw,
            tuple(arrow["cell"]),
            arrow["direction"],
            image_size,
            grid_size,
            color=(8, 8, 8),
            width=line_width,
            length_scale=0.62,
            head_scale=0.26,
        )

    draw_start_marker(draw, tuple(sample["start"]), image_size, grid_size)
    cls = classes[sample["label"]]
    draw_shape(draw, tuple(sample["end"]), cls.shape, cls.rgb, image_size, grid_size)
    return image


def split_config(config: dict, key: str, split: str) -> dict:
    base = dict(config.get(key, {}))
    overrides = config.get(f"split_{key}", {}).get(split, {})
    base.update(overrides)
    return base


def make_sample(rng: random.Random, config: dict, classes: Sequence[VSTDClass], split: str, index: int) -> dict:
    factors = split_config(config, "factors", split)
    generation = split_config(config, "generation", split)
    grid_size = int(config["image"]["grid_size"])
    path_length = int(rng.choice(factors["path_lengths"]))
    distractor_ratio = float(rng.choice(factors["distractor_ratios"]))
    path = random_path(
        rng,
        grid_size=grid_size,
        path_length=path_length,
        min_turns=int(generation.get("min_turns", 1)),
        max_attempts=int(generation.get("max_attempts_per_sample", 200)),
        start=tuple(generation["fixed_start"]) if "fixed_start" in generation else None,
        min_manhattan=int(generation.get("min_manhattan", 0)),
    )
    label_class = rng.choice(classes)
    arrows = [
        {"cell": list(path[i]), "direction": direction_between(path[i], path[i + 1])}
        for i in range(len(path) - 1)
    ]
    metrics = path_metrics(path, grid_size)
    decoy_items = sample_decoy_paths(rng, config, generation, grid_size, path, classes, [])
    decoy_cells = [tuple(item["cell"]) for item in decoy_items]
    distractors = sample_distractors(
        rng,
        grid_size=grid_size,
        true_path=path,
        ratio=distractor_ratio,
        classes=classes,
        allow_shapes=bool(generation.get("distractor_shapes", True)),
        black_arrow_prob=float(generation.get("distractor_black_prob", 0.0)),
        extra_occupied=decoy_cells,
    )
    distractors.extend(decoy_items)
    decoy_arrow_count = sum(1 for item in distractors if item.get("decoy_path") and item["type"] == "arrow")
    filename = f"{index:06d}.png"
    return {
        "image": f"{split}/{filename}",
        "split": split,
        "label": label_class.id,
        "class_name": label_class.name,
        "prompt": label_class.prompt,
        "path_length": path_length,
        "distractor_ratio": distractor_ratio,
        "decoy_arrow_count": decoy_arrow_count,
        "grid_size": grid_size,
        "image_size": int(config["image"]["size"]),
        "start": list(path[0]),
        "end": list(path[-1]),
        "path": [list(cell) for cell in path],
        **metrics,
        "arrows": arrows,
        "distractors": distractors,
    }


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def generate_dataset(config_path: Path, output_root: Path, overwrite: bool = False) -> Path:
    config = load_config(config_path)
    dataset_name = config["dataset_name"]
    out_dir = output_root / dataset_name
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, out_dir / "dataset_config.yaml")
    classes = build_classes(config)
    write_json(
        out_dir / "classes.json",
        [
            {
                "id": cls.id,
                "name": cls.name,
                "prompt": cls.prompt,
                "color": cls.color,
                "shape": cls.shape,
                "rgb": list(cls.rgb),
            }
            for cls in classes
        ],
    )

    rng = random.Random(int(config["seed"]))
    metadata_path = out_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as meta_f:
        for split, count in config["splits"].items():
            split_dir = out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(int(count)):
                sample = make_sample(rng, config, classes, split, idx)
                image = render_sample(config, classes, sample)
                image.save(out_dir / sample["image"])
                meta_f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return out_dir
