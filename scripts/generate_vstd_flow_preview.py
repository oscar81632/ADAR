#!/usr/bin/env python3
"""Generate a small exploratory VSTD-Flow preview dataset.

The images are flowchart-like variants of VSTD. They are intended for visual
inspection first, not for thesis-main experiments.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "red": (214, 65, 61),
    "blue": (57, 113, 202),
    "green": (68, 150, 97),
    "yellow": (224, 169, 43),
    "purple": (124, 84, 190),
    "gray": (118, 128, 140),
    "ink": (31, 35, 43),
    "paper": (250, 250, 248),
    "line": (64, 69, 78),
    "decoy": (154, 160, 170),
}


ENDPOINTS = [
    ("red", "output"),
    ("blue", "output"),
    ("green", "database"),
    ("yellow", "database"),
    ("red", "module"),
    ("blue", "module"),
    ("green", "service"),
    ("yellow", "service"),
]


def load_font(size: int, bold: bool = False):
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = load_font(18)
FONT_SMALL = load_font(14)
FONT_BOLD = load_font(18, bold=True)


def class_records():
    records = []
    for idx, (color, kind) in enumerate(ENDPOINTS):
        name = f"{color}_{kind}"
        records.append(
            {
                "id": idx,
                "name": name,
                "prompt": f"the flow ends at the {color} {kind} node",
                "color": color,
                "shape": kind,
            }
        )
    return records


def jitter(point, rng, amount=10):
    return (point[0] + rng.randint(-amount, amount), point[1] + rng.randint(-amount, amount))


def draw_shadow_box(draw, box, fill, outline, radius=12, width=2):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 3, y0 + 4, x1 + 3, y1 + 4), radius=radius, fill=(220, 224, 228))
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_diamond(draw, center, size, fill, outline, width=2):
    x, y = center
    w, h = size
    points = [(x, y - h // 2), (x + w // 2, y), (x, y + h // 2), (x - w // 2, y)]
    draw.polygon([(px + 3, py + 4) for px, py in points], fill=(220, 224, 228))
    draw.polygon(points, fill=fill, outline=outline)
    draw.line(points + [points[0]], fill=outline, width=width, joint="curve")


def draw_cylinder(draw, box, fill, outline, width=2):
    x0, y0, x1, y1 = box
    h = 18
    draw.rounded_rectangle((x0 + 3, y0 + 4, x1 + 3, y1 + 4), radius=10, fill=(220, 224, 228))
    draw.rectangle((x0, y0 + h // 2, x1, y1 - h // 2), fill=fill)
    draw.ellipse((x0, y0, x1, y0 + h), fill=fill, outline=outline, width=width)
    draw.ellipse((x0, y1 - h, x1, y1), fill=fill, outline=outline, width=width)
    draw.line((x0, y0 + h // 2, x0, y1 - h // 2), fill=outline, width=width)
    draw.line((x1, y0 + h // 2, x1, y1 - h // 2), fill=outline, width=width)


def label(draw, center, text, font=FONT, fill=COLORS["ink"]):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((center[0] - (bbox[2] - bbox[0]) / 2, center[1] - (bbox[3] - bbox[1]) / 2), text, font=font, fill=fill)


def draw_node(draw, node, active=True):
    x, y = node["pos"]
    kind = node["kind"]
    color_name = node.get("color", "gray")
    fill = tuple(min(255, int(c * 0.18 + 255 * 0.82)) for c in COLORS[color_name])
    outline = COLORS[color_name] if active else COLORS["decoy"]
    text_fill = COLORS["ink"] if active else (105, 110, 118)
    if kind == "decision":
        draw_diamond(draw, (x, y), (96, 70), fill=fill, outline=outline, width=3 if active else 2)
    elif kind == "database":
        draw_cylinder(draw, (x - 52, y - 31, x + 52, y + 31), fill=fill, outline=outline, width=3 if active else 2)
    elif kind == "output":
        draw_shadow_box(draw, (x - 56, y - 30, x + 56, y + 30), fill=fill, outline=outline, radius=20, width=3 if active else 2)
    else:
        draw_shadow_box(draw, (x - 56, y - 32, x + 56, y + 32), fill=fill, outline=outline, radius=10, width=3 if active else 2)
    label(draw, (x, y), node["text"], font=FONT_BOLD if active else FONT_SMALL, fill=text_fill)


def node_box(node):
    x, y = node["pos"]
    kind = node["kind"]
    if kind == "decision":
        return (x - 52, y - 38, x + 52, y + 38)
    if kind == "database":
        return (x - 56, y - 35, x + 56, y + 35)
    if kind == "output":
        return (x - 60, y - 34, x + 60, y + 34)
    return (x - 60, y - 36, x + 60, y + 36)


def port(node, side, pad=8):
    x0, y0, x1, y1 = node_box(node)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    if side == "right":
        return (x1 + pad, cy)
    if side == "left":
        return (x0 - pad, cy)
    if side == "top":
        return (cx, y0 - pad)
    if side == "bottom":
        return (cx, y1 + pad)
    raise ValueError(f"Unknown side: {side}")


def draw_arrow_head(draw, start, end, fill, width=4):
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 13 if width >= 4 else 10
    tip = (ex, ey)
    base = (ex - ux * size, ey - uy * size)
    points = [
        tip,
        (base[0] + px * size * 0.48, base[1] + py * size * 0.48),
        (base[0] - px * size * 0.48, base[1] - py * size * 0.48),
    ]
    draw.polygon(points, fill=fill)


def draw_dashed_segment(draw, start, end, fill, width=3, dash_length=12, gap=8):
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    distance = 0.0
    while distance < length:
        a = distance
        b = min(length, distance + dash_length)
        draw.line((sx + ux * a, sy + uy * a, sx + ux * b, sy + uy * b), fill=fill, width=width)
        distance += dash_length + gap


def draw_connector(draw, points, fill, width=4, dash=False):
    cleaned = [points[0]]
    for point in points[1:]:
        if abs(point[0] - cleaned[-1][0]) > 1 or abs(point[1] - cleaned[-1][1]) > 1:
            cleaned.append(point)
    for start, end in zip(cleaned[:-1], cleaned[1:]):
        if dash:
            draw_dashed_segment(draw, start, end, fill=fill, width=width)
        else:
            draw.line((*start, *end), fill=fill, width=width)
    if len(cleaned) >= 2:
        draw_arrow_head(draw, cleaned[-2], cleaned[-1], fill=fill, width=width)


def main_route(a, b):
    start = port(a, "right")
    end = port(b, "left")
    mid_x = (start[0] + end[0]) / 2
    return [start, (mid_x, start[1]), (mid_x, end[1]), end]


def decoy_route(a, b):
    ax, ay = a["pos"]
    bx, by = b["pos"]
    if by < ay:
        start = port(a, "top")
        end = port(b, "bottom")
    elif by > ay:
        start = port(a, "bottom")
        end = port(b, "top")
    elif bx > ax:
        start = port(a, "right")
        end = port(b, "left")
    else:
        start = port(a, "left")
        end = port(b, "right")

    if abs(start[0] - end[0]) > abs(start[1] - end[1]):
        mid_x = (start[0] + end[0]) / 2
        return [start, (mid_x, start[1]), (mid_x, end[1]), end]
    mid_y = (start[1] + end[1]) / 2
    return [start, (start[0], mid_y), (end[0], mid_y), end]


def make_sample(rng, index, out_dir, split="test"):
    label_id = rng.randrange(len(ENDPOINTS))
    color, endpoint_kind = ENDPOINTS[label_id]
    bg = Image.new("RGB", (768, 512), COLORS["paper"])
    draw = ImageDraw.Draw(bg)

    for x in range(48, 768, 64):
        draw.line((x, 36, x, 476), fill=(235, 237, 239), width=1)
    for y in range(48, 512, 64):
        draw.line((36, y, 732, y), fill=(235, 237, 239), width=1)

    title = f"Workflow Trace #{index:03d}"
    draw.text((28, 18), title, font=FONT_SMALL, fill=(95, 102, 112))

    main_positions = [
        (72, 256),
        jitter((235, 256), rng, 10),
        jitter((380, rng.choice([176, 256, 336])), rng, 10),
        jitter((525, rng.choice([154, 256, 358])), rng, 10),
        (680, rng.choice([148, 256, 364])),
    ]
    kinds = ["start", "module", rng.choice(["decision", "module"]), rng.choice(["module", "database"]), endpoint_kind]
    texts = ["START", "Parse", "Route" if kinds[2] == "decision" else "Check", "Transform" if kinds[3] == "module" else "Store", color.upper()]
    main_nodes = [
        {"pos": pos, "kind": kind, "text": text, "color": "purple" if idx == 0 else color if idx == len(main_positions) - 1 else "blue"}
        for idx, (pos, kind, text) in enumerate(zip(main_positions, kinds, texts))
    ]

    decoy_nodes = []
    decoy_targets = rng.sample([i for i in range(len(ENDPOINTS)) if i != label_id], 3)
    candidate_pos = [(236, 96), (380, 84), (528, 92), (252, 420), (424, 424), (596, 430)]
    rng.shuffle(candidate_pos)
    for i, target_id in enumerate(decoy_targets):
        d_color, d_kind = ENDPOINTS[target_id]
        decoy_nodes.append(
            {
                "pos": jitter(candidate_pos[i], rng, 10),
                "kind": d_kind if i % 2 else "module",
                "text": d_color[:3].upper(),
                "color": d_color,
            }
        )

    decoy_edges = [
        (main_nodes[1], decoy_nodes[0]),
        (decoy_nodes[0], decoy_nodes[1]),
        (main_nodes[2], decoy_nodes[2]),
    ]

    for a, b in decoy_edges:
        draw_connector(draw, decoy_route(a, b), fill=COLORS["decoy"], width=3, dash=True)

    for node in decoy_nodes:
        draw_node(draw, node, active=False)
    for node in main_nodes:
        draw_node(draw, node, active=True)

    for a, b in zip(main_nodes[:-1], main_nodes[1:]):
        draw_connector(draw, main_route(a, b), fill=COLORS["purple"], width=5, dash=False)

    image_name = f"images/{split}_{index:04d}.png"
    image_path = out_dir / image_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    bg.save(image_path)
    prompt = f"the flow ends at the {color} {endpoint_kind} node"
    return {
        "split": split,
        "index": index,
        "image": image_name,
        "label": label_id,
        "class_name": f"{color}_{endpoint_kind}",
        "prompt": prompt,
        "path_length": len(main_nodes) - 1,
        "flow_nodes": len(main_nodes) + len(decoy_nodes),
        "decoy_branch_count": len(decoy_edges),
        "style": "flowchart_preview",
    }


def make_contact_sheet(out_dir, rows, columns=3):
    images = [Image.open(out_dir / row["image"]).convert("RGB") for row in rows]
    thumb_w, thumb_h = 384, 256
    sheet_h = ((len(images) + columns - 1) // columns) * thumb_h
    sheet = Image.new("RGB", (columns * thumb_w, sheet_h), (245, 246, 248))
    for idx, image in enumerate(images):
        image = image.resize((thumb_w, thumb_h), Image.BICUBIC)
        x = (idx % columns) * thumb_w
        y = (idx // columns) * thumb_h
        sheet.paste(image, (x, y))
    sheet.save(out_dir / "preview_grid.png")


def main():
    parser = argparse.ArgumentParser(description="Generate VSTD-Flow preview images.")
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/exploratory_vstd_flow_preview"))
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    classes = class_records()
    with (args.output_dir / "classes.json").open("w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    rows = [make_sample(rng, i, args.output_dir, split="test") for i in range(args.num_samples)]
    with (args.output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    make_contact_sheet(args.output_dir, rows)
    print(f"Wrote {len(rows)} preview images to {args.output_dir}")
    print(f"Contact sheet: {args.output_dir / 'preview_grid.png'}")


if __name__ == "__main__":
    main()
