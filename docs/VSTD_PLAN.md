# Visual State Tracking Dataset Plan

## Goal

Create a controlled benchmark where the label depends on following a visual path
through a grid. The task is designed to test long-range visual dependency and
state-like propagation, rather than ordinary object recognition.

## Dataset Idea

Each image contains:

- A start marker.
- A path made of arrows.
- A final colored shape.
- Distractor arrows or shapes that do not belong to the true path.

The label is the object reached by following the path from the start marker.

Example prompt classes:

```text
a path ending at a red circle
a path ending at a blue square
a path ending at a green triangle
```

## Controlled Factors

| Factor | Values |
|---|---|
| Grid size | 14x14, 16x16, 24x24 |
| Path length | 4, 8, 12, 16, 24 |
| Distractor ratio | 0.0, 0.25, 0.5, 0.75 |
| Image size | 224, 384, 512 |
| Class count | 8 or 12 |

## Main Hypothesis

Vim should become more competitive when the task requires longer visual state
tracking, especially as path length and distractor density increase.

## Current Experimental Scope

The first version intentionally keeps the visual token lattice fixed:

```text
image size = 224 x 224
patch size = 16 x 16
grid size = 14 x 14
visual tokens = 196
```

This means the first benchmark does **not** claim token sequence length scaling.
Instead, it studies long dependency paths over a fixed visual token grid.

The intended claim is:

> Under the same input resolution and patch lattice, classification becomes
> harder as the number of required visual state transitions increases.

High-resolution token-count scaling can be added later, but it requires careful
positional-embedding interpolation for both ViT and Vim.

## Long-Dependency Metadata

Each generated sample stores:

| Field | Meaning |
|---|---|
| `path_length` | Number of arrow-following steps from start to end. |
| `start_end_manhattan` | Manhattan distance between start and end cells. |
| `num_turns` | Number of direction changes along the true path. |
| `path_bbox_area` | Area of the bounding box covered by the path. |
| `path_coverage` | Fraction of grid cells used by the path. |
| `long_dependency_score` | Weighted summary of length, distance, and spatial extent. |

`path_length` alone is not enough to prove long-range dependency. The analysis
should also report accuracy against distance, turns, bounding-box area, and the
combined score.

## Required Baselines

- ViT-CLIP zero-shot or fine-tuned.
- Vim-CLIP fine-tuned.
- Vim adapter fine-tuned.
- Vim LoRA fine-tuned.

## Metrics

- Accuracy.
- Accuracy by path length.
- Accuracy by start-end Manhattan distance.
- Accuracy by path bounding-box area.
- Accuracy by long-dependency score.
- Accuracy by distractor ratio.
- Inference latency and throughput.
- Similarity gap between correct and strongest incorrect text class.
- Prototype similarity and PCA spectrum.
