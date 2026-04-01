"""
Generate PCB harmonization dataset for qwen-image-finetune.

Uses OminiControl's on-the-fly composite building logic:
1. Random crop from board → target image (real board)
2. Build composite by pasting matched components → control image
3. Generate editing prompt from 3 selected templates
4. Generate edit mask (white areas = edit region)

Output format:
    data/pcb_harmonize/
        training_images/
            {board}_{idx}.png       ← target (real board crop)
            {board}_{idx}.txt       ← prompt
        control_images/
            {board}_{idx}.png       ← control (composite)
            {board}_{idx}_mask.png  ← edit mask (white=edit region)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add OminiControl to path for ComponentBankV2
sys.path.insert(0, "/home/xinrui/projects/OminiControl")
from lib.component_bank_v2 import ComponentBankV2, get_annotations_in_crop, CAT_ID_TO_NAME


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


def build_composite(
    annotations: list,
    bank: ComponentBankV2,
    exclude_board: str,
    board_color: str,
    width: int,
    height: int,
    resize_jitter: float = 0.15,
) -> Image.Image:
    """Build a composite image by pasting matched components onto white canvas."""
    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    for ann in annotations:
        cat_name = ann["category_name"]
        rx, ry, rw, rh = ann["bbox"]
        rw, rh = int(rw), int(rh)

        if rw < 3 or rh < 3:
            continue

        match = bank.find_match(
            category=cat_name,
            target_w=rw,
            target_h=rh,
            board_color=board_color,
            exclude_board=exclude_board,
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


def build_edit_mask(composite: Image.Image, threshold: int = 250) -> Image.Image:
    """Generate edit mask: white areas (background) = 255, component areas = 0.

    In Qwen edit_mask_loss: foreground(1) = edit region → higher weight.
    So white background → 255 (edit), components → 0 (preserve).
    """
    arr = np.array(composite)
    # White pixels: all channels > threshold
    is_white = np.all(arr > threshold, axis=2)
    mask = np.where(is_white, 255, 0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_dir", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/annotation/train")
    parser.add_argument("--image_dir", default="/home/xinrui/projects/data/ti_pcb/layout_data/v2_Color_Res_Class_xywh/image/train")
    parser.add_argument("--output_dir", default="/home/xinrui/projects/Qwen-Image/qwen-image-finetune/data/pcb_harmonize")
    parser.add_argument("--crop_size", type=int, default=512,
                        help="Crop size from board (will be upscaled to target_size)")
    parser.add_argument("--target_size", type=int, default=1024,
                        help="Final output resolution")
    parser.add_argument("--crops_per_board", type=int, default=10)
    parser.add_argument("--zoom_prob", type=float, default=0.4,
                        help="Probability of zoom crop (crop_size/2 → crop_size)")
    parser.add_argument("--resize_jitter", type=float, default=0.15)
    parser.add_argument("--min_components", type=int, default=2)
    parser.add_argument("--min_visible_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max total samples to generate (-1 = all)")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    training_dir = output_dir / "training_images"
    control_dir = output_dir / "control_images"
    training_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ComponentBankV2
    print("Loading ComponentBankV2...")
    bank = ComponentBankV2(
        anno_dir=args.anno_dir,
        image_dir=args.image_dir,
    )

    # Load board list
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

        boards.append({
            "name": board_name,
            "color": data.get("board_color", "green"),
            "annotations": data.get("annotations", []),
        })

    print(f"Loaded {len(boards)} boards")

    # Category ID → name mapping
    # Load from first annotation file
    cat_map = {}
    sample_anno = os.path.join(args.anno_dir, os.listdir(args.anno_dir)[0])
    with open(sample_anno) as f:
        d = json.load(f)
        for cat in d.get("categories", []):
            cat_map[cat["id"]] = cat["name"]

    total_generated = 0
    crop_size = args.crop_size
    half_crop = crop_size // 2  # For zoom crops

    for board in tqdm(boards, desc="Boards"):
        board_name = board["name"]
        board_color = board["color"]
        all_annotations = board["annotations"]

        # Add category_name to annotations
        for ann in all_annotations:
            if "category_name" not in ann:
                ann["category_name"] = cat_map.get(ann["category_id"], "unknown")

        img_path = os.path.join(args.image_dir, f"{board_name}.png")
        board_img = Image.open(img_path).convert("RGB")
        img_w, img_h = board_img.size

        for crop_idx in range(args.crops_per_board):
            if args.max_samples > 0 and total_generated >= args.max_samples:
                break

            # Decide crop size: zoom or normal
            use_zoom = random.random() < args.zoom_prob
            actual_crop = half_crop if use_zoom else crop_size

            # Try up to 5 random positions to find enough components
            found = False
            for _ in range(5):
                cx = random.randint(0, max(0, img_w - actual_crop))
                cy = random.randint(0, max(0, img_h - actual_crop))
                crop_annotations = get_annotations_in_crop(
                    all_annotations, cx, cy, actual_crop, args.min_visible_ratio
                )
                if len(crop_annotations) >= args.min_components:
                    found = True
                    break

            if not found:
                continue

            # Real patch (target)
            real_patch = board_img.crop((cx, cy, cx + actual_crop, cy + actual_crop))

            if use_zoom:
                # Upscale annotations to crop_size space
                scale = crop_size / actual_crop
                real_patch = real_patch.resize((crop_size, crop_size), Image.LANCZOS)
                crop_annotations = [
                    {**ann, "bbox": (
                        ann["bbox"][0] * scale, ann["bbox"][1] * scale,
                        ann["bbox"][2] * scale, ann["bbox"][3] * scale
                    )}
                    for ann in crop_annotations
                ]

            # Build composite
            composite = build_composite(
                crop_annotations, bank, board_name, board_color,
                crop_size, crop_size, args.resize_jitter
            )

            # Upscale to target resolution if needed
            if args.target_size != crop_size:
                real_patch = real_patch.resize(
                    (args.target_size, args.target_size), Image.LANCZOS
                )
                composite = composite.resize(
                    (args.target_size, args.target_size), Image.LANCZOS
                )

            # Generate edit mask (before upscale for accuracy, then upscale)
            # Actually generate from the final composite
            edit_mask = build_edit_mask(composite)

            # Generate prompt
            prompt = make_prompt(board_color)

            # Save
            sample_id = f"{board_name}_{crop_idx:03d}"

            real_patch.save(training_dir / f"{sample_id}.png")
            composite.save(control_dir / f"{sample_id}.png")
            edit_mask.save(control_dir / f"{sample_id}_mask.png")

            with open(training_dir / f"{sample_id}.txt", "w") as f:
                f.write(prompt)

            total_generated += 1

        if args.max_samples > 0 and total_generated >= args.max_samples:
            break

    print(f"\nDone! Generated {total_generated} samples")
    print(f"  Training images: {training_dir}")
    print(f"  Control images:  {control_dir}")
    print(f"  Output dir:      {output_dir}")


if __name__ == "__main__":
    main()
