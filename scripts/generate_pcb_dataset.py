"""
Generate PCB harmonization dataset for qwen-image-finetune.

Uses OminiControl's on-the-fly composite building logic with multiprocessing:
1. Random crop from board → target image (real board)
2. Build composite by pasting matched components → control image
3. Generate editing prompt from 3 selected templates
4. Generate edit mask (white areas = edit region)

Output format:
    PCB_harmonize/
        training_images/
            {board}_{idx}.png       ← target (real board crop)
            {board}_{idx}.txt       ← prompt
        control_images/
            {board}_{idx}.png       ← control (composite)
            {board}_{idx}_mask.png  ← edit mask (white=edit region)

Usage:
    python generate_pcb_dataset.py --workers 16
"""

import argparse
import json
import os
import random
import sys
from multiprocessing import Pool, current_process
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add OminiControl to path for ComponentBankV2
OMINI_PATH = os.environ.get("OMINI_PATH", "/home/xinrui/projects/OminiControl")
sys.path.insert(0, OMINI_PATH)
try:
    from lib.component_bank_v2 import ComponentBankV2, get_annotations_in_crop, CAT_ID_TO_NAME
except ModuleNotFoundError:
    from component_bank_v2 import ComponentBankV2, get_annotations_in_crop, CAT_ID_TO_NAME


# ---------------------------------------------------------------------------
# Prompt templates — editing instructions for Qwen-Image-Edit
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = [
    # T8_silkscreen
    (
        "The electronic components in this image are pasted onto a white background. "
        "Place them onto a {color} PCB board — generate the board surface, copper traces, "
        "silkscreen markings, and solder connections while keeping every component exactly where it is."
    ),
    # T8_silk_detail
    (
        "The electronic components in this image are pasted onto a white background. "
        "Place them onto a {color} PCB board — generate the board surface with copper traces, "
        "solder pads, vias, and white silkscreen labels near each component. "
        "Keep every component exactly where it is."
    ),
    # T11_inpaint_silk
    (
        "Inpaint the white/empty regions with a {color} PCB board surface. "
        "Add copper traces, solder pads, and white silkscreen labels. "
        "Keep all electronic components unchanged."
    ),
]

COLOR_NAMES = {
    "green": "green", "red": "red", "blue": "blue",
    "black": "black", "white": "white", "yellow": "yellow",
}


def make_prompt(board_color: str) -> str:
    color = COLOR_NAMES.get(board_color, "green")
    template = random.choice(PROMPT_TEMPLATES)
    return template.format(color=color)


def build_composite(annotations, bank, exclude_board, board_color, width, height, resize_jitter=0.15):
    """Build a composite image by pasting matched components onto white canvas."""
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    for ann in annotations:
        cat_name = ann["category_name"]
        rx, ry, rw, rh = ann["bbox"]
        rw, rh = int(rw), int(rh)
        if rw < 3 or rh < 3:
            continue
        match = bank.find_match(
            category=cat_name, target_w=rw, target_h=rh,
            board_color=board_color, exclude_board=exclude_board,
        )
        if match is None:
            continue
        crop = bank.load_crop(match, rw, rh, resize_jitter=resize_jitter)
        if crop is None:
            continue
        px = max(0, min(int(rx), width - crop.width))
        py = max(0, min(int(ry), height - crop.height))
        canvas.paste(crop, (px, py))
    return canvas


def build_edit_mask(composite, threshold=250):
    """Generate edit mask: white areas = 255 (edit region), component areas = 0 (preserve)."""
    arr = np.array(composite)
    is_white = np.all(arr > threshold, axis=2)
    mask = np.where(is_white, 255, 0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------
def process_board_chunk(args_tuple):
    """Process a chunk of boards. Each worker gets its own ComponentBankV2."""
    board_chunk, config = args_tuple
    worker_id = current_process().name

    # Each worker creates its own ComponentBankV2 (not picklable)
    bank = ComponentBankV2(
        anno_dir=config["anno_dir"],
        image_dir=config["image_dir"],
    )

    training_dir = Path(config["output_dir"]) / "training_images"
    control_dir = Path(config["output_dir"]) / "control_images"

    crop_size = config["crop_size"]
    half_crop = crop_size // 2
    target_size = config["target_size"]
    generated = 0

    for board in board_chunk:
        board_name = board["name"]
        board_color = board["color"]
        all_annotations = board["annotations"]

        img_path = os.path.join(config["image_dir"], f"{board_name}.png")
        if not os.path.exists(img_path):
            continue
        board_img = Image.open(img_path).convert("RGB")
        img_w, img_h = board_img.size

        for crop_idx in range(config["crops_per_board"]):
            # Deterministic seed per sample for reproducibility
            sample_seed = hash(f"{board_name}_{crop_idx}") % (2**31)
            random.seed(sample_seed)

            use_zoom = random.random() < config["zoom_prob"]
            actual_crop = half_crop if use_zoom else crop_size

            found = False
            for _ in range(5):
                cx = random.randint(0, max(0, img_w - actual_crop))
                cy = random.randint(0, max(0, img_h - actual_crop))
                crop_annotations = get_annotations_in_crop(
                    all_annotations, cx, cy, actual_crop, config["min_visible_ratio"]
                )
                if len(crop_annotations) >= config["min_components"]:
                    found = True
                    break

            if not found:
                continue

            real_patch = board_img.crop((cx, cy, cx + actual_crop, cy + actual_crop))

            if use_zoom:
                scale = crop_size / actual_crop
                real_patch = real_patch.resize((crop_size, crop_size), Image.LANCZOS)
                crop_annotations = [
                    {**ann, "bbox": (
                        ann["bbox"][0] * scale, ann["bbox"][1] * scale,
                        ann["bbox"][2] * scale, ann["bbox"][3] * scale
                    )}
                    for ann in crop_annotations
                ]

            composite = build_composite(
                crop_annotations, bank, board_name, board_color,
                crop_size, crop_size, config["resize_jitter"]
            )

            if target_size != crop_size:
                real_patch = real_patch.resize((target_size, target_size), Image.LANCZOS)
                composite = composite.resize((target_size, target_size), Image.LANCZOS)

            edit_mask = build_edit_mask(composite)
            prompt = make_prompt(board_color)

            sample_id = f"{board_name}_{crop_idx:03d}"
            real_patch.save(training_dir / f"{sample_id}.png")
            composite.save(control_dir / f"{sample_id}.png")
            edit_mask.save(control_dir / f"{sample_id}_mask.png")
            with open(training_dir / f"{sample_id}.txt", "w") as f:
                f.write(prompt)

            generated += 1

    return generated


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_dir",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/annotation/train")
    parser.add_argument("--image_dir",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/train")
    parser.add_argument("--output_dir",
        default="/projects/_ssd/xrssd/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/PCB_harmonize")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--target_size", type=int, default=1024)
    parser.add_argument("--crops_per_board", type=int, default=10)
    parser.add_argument("--zoom_prob", type=float, default=0.4)
    parser.add_argument("--resize_jitter", type=float, default=0.15)
    parser.add_argument("--min_components", type=int, default=2)
    parser.add_argument("--min_visible_ratio", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--max_samples", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    training_dir = output_dir / "training_images"
    control_dir = output_dir / "control_images"
    training_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)

    # Load board list with annotations
    print("Loading board list...")
    # Get category map from first annotation
    cat_map = {}
    sample_anno = os.path.join(args.anno_dir, sorted(os.listdir(args.anno_dir))[0])
    with open(sample_anno) as f:
        for cat in json.load(f).get("categories", []):
            cat_map[cat["id"]] = cat["name"]

    boards = []
    for f in sorted(os.listdir(args.anno_dir)):
        if not f.endswith(".json"):
            continue
        board_name = f[:-5]
        img_path = os.path.join(args.image_dir, f"{board_name}.png")
        if not os.path.exists(img_path):
            continue
        with open(os.path.join(args.anno_dir, f)) as fh:
            data = json.load(fh)
        annotations = data.get("annotations", [])
        for ann in annotations:
            if "category_name" not in ann:
                ann["category_name"] = cat_map.get(ann["category_id"], "unknown")
        boards.append({
            "name": board_name,
            "color": data.get("board_color", "green"),
            "annotations": annotations,
        })

    if args.max_samples > 0:
        max_boards = args.max_samples // args.crops_per_board + 1
        boards = boards[:max_boards]

    print(f"Loaded {len(boards)} boards, generating {args.crops_per_board} crops each")
    print(f"Workers: {args.workers}, Target size: {args.target_size}")
    print(f"Output: {args.output_dir}")

    # Split boards into chunks for workers
    config = {
        "anno_dir": args.anno_dir,
        "image_dir": args.image_dir,
        "output_dir": str(args.output_dir),
        "crop_size": args.crop_size,
        "target_size": args.target_size,
        "crops_per_board": args.crops_per_board,
        "zoom_prob": args.zoom_prob,
        "resize_jitter": args.resize_jitter,
        "min_components": args.min_components,
        "min_visible_ratio": args.min_visible_ratio,
    }

    chunk_size = max(1, len(boards) // args.workers)
    chunks = []
    for i in range(0, len(boards), chunk_size):
        chunks.append((boards[i:i + chunk_size], config))

    print(f"Split into {len(chunks)} chunks")

    # Run with multiprocessing
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_board_chunk, chunks),
                total=len(chunks), desc="Chunks"
            ))
    else:
        results = [process_board_chunk(chunks[0])]

    total = sum(results)
    print(f"\nDone! Generated {total} samples")
    print(f"  Training images: {training_dir}")
    print(f"  Control images:  {control_dir}")


if __name__ == "__main__":
    main()
