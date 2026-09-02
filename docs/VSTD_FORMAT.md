# VSTD File Format

Generated datasets are stored under `datasets/<dataset_name>/`.

```text
datasets/<dataset_name>/
  dataset_config.yaml
  classes.json
  metadata.jsonl
  train/
    000000.png
    ...
  val/
    000000.png
  test/
    000000.png
  previews/
    sample_grid.png
```

## Classes

Classes are color-shape pairs. For example:

```json
{
  "id": 0,
  "name": "red circle",
  "prompt": "a path ending at a red circle",
  "color": "red",
  "shape": "circle"
}
```

## Metadata JSONL

Each line in `metadata.jsonl` contains one sample:

```json
{
  "image": "train/000000.png",
  "split": "train",
  "label": 0,
  "class_name": "red circle",
  "prompt": "a path ending at a red circle",
  "path_length": 8,
  "distractor_ratio": 0.25,
  "grid_size": 14,
  "image_size": 224,
  "start": [2, 3],
  "end": [9, 7],
  "path": [[2, 3], [3, 3], [4, 3], "..."],
  "arrows": [
    {"cell": [2, 3], "direction": "down"}
  ],
  "distractors": [
    {"cell": [5, 10], "type": "arrow", "direction": "left"}
  ]
}
```

The label is determined only by the object at the final path cell. Distractors
are sampled away from the true path whenever possible.
