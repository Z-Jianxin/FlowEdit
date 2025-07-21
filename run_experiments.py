#!/usr/bin/env python3
"""
Run FlowEdit semantic‑editing experiments defined in a JSON dataset file.

The implementation is a near‑verbatim copy of the official Gradio demo
to guarantee that **the same seed ⇒ the same edited image bytes**.

Metrics reported per example
  • CLIP similarity (target prompt ↔ edited image)
  • CLIP similarity (target prompt ↔ source image)
  • LPIPS distance (edited image ↔ source image)
  • Pixelwise L1 mean & median
  • Pixelwise L2 mean & median
  • Paths to edited & diff images

Results are printed to stdout and also written to a CSV file.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageChops
from transformers import CLIPModel, CLIPProcessor

# FlowEdit imports (identical to the demo)
from diffusers import FluxPipeline
from FlowEdit_utils import FlowEditFLUX

import lpips

# ------------------------------------------------------------
# 1.  Metric helpers  (unchanged)
# ------------------------------------------------------------
class MetricComputer:
    def __init__(self, device: torch.device):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
        self.lpips_transform = T.Compose(
            [T.Resize((1024, 1024)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]
        )

    @torch.inference_mode()
    def clip_similarity(self, image: Image.Image, text: str) -> float:
        inputs = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True
                                     ).to(self.clip_model.device)
        return self.clip_model(**inputs).logits_per_image.item()

    @torch.inference_mode()
    def lpips_distance(self, img1: Image.Image, img2: Image.Image) -> float:
        t1 = self.lpips_transform(img1).unsqueeze(0).to(self.device)
        t2 = self.lpips_transform(img2).unsqueeze(0).to(self.device)
        return self.lpips_fn(t1, t2).item()

    @staticmethod
    def pixel_distances(img1: Image.Image, img2: Image.Image) -> Tuple[float, float, float, float]:
        a1 = np.asarray(img1, dtype=np.float32)
        a2 = np.asarray(img2, dtype=np.float32)
        # per‑channel difference
        diff = a1 - a2
        # average across the 3 channels so l1/l2 match image (H, W)
        l1 = np.mean(np.abs(diff), axis=-1)      # shape (H, W)
        l2 = np.mean(diff ** 2,  axis=-1)        # shape (H, W)
        return l1.mean(), np.median(l1), l2.mean(), np.median(l2)


# ------------------------------------------------------------
# 2.  FlowEdit “FluxEditor”  (copied almost verbatim)
# ------------------------------------------------------------
class FluxEditor:
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16
        )  # moved to device inside .edit

        # lightweight CLIP for demo‐style printouts (optional)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.lpips = lpips.LPIPS(net="vgg").to(self.device)
        self.lpips_transform = T.Compose(
            [T.Resize((1024, 1024)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]
        )

    def print_clip_score(self, image: Image.Image, prompt: str):
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        out = self.clip_model(**inputs)
        print("CLIP score:", out.logits_per_image.item())

    # --------------------------------------------------------
    #  core editing logic copied from the Gradio demo
    # --------------------------------------------------------
    @torch.inference_mode()
    def edit(
        self,
        image: Image.Image,
        source_prompt: str,
        target_prompt: str,
        negative_prompt: str | None,
        T_steps: int,
        n_avg: int,
        source_guidance_scale: float,
        target_guidance_scale: float,
        n_min: int,
        n_max: int,
        seed: int,
    ):
        if not seed:
            seed = int(torch.Generator(device="cpu").seed())
        print("Random seed:", seed)
        torch.manual_seed(seed)

        device = self.device
        self.pipe = self.pipe.to(device)

        negative_prompt = negative_prompt or None

        # ---------------- image -> latent ----------------
        img_src = self.pipe.image_processor.preprocess(image, height=1024, width=1024).to(device).half()
        with torch.autocast("cuda"), torch.inference_mode():
            lat_src_denorm = self.pipe.vae.encode(img_src).latent_dist.mode()
        lat_src = (lat_src_denorm - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor

        # ---------------- FlowEdit main call -------------
        lat_tar = FlowEditFLUX(
            self.pipe,
            self.pipe.scheduler,
            lat_src,
            source_prompt,
            target_prompt,
            negative_prompt,
            T_steps,
            n_avg,
            source_guidance_scale,
            target_guidance_scale,
            n_min,
            n_max,
        )

        # -------------- latent -> image ------------------
        lat_tar_denorm = lat_tar / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            img_tar = self.pipe.vae.decode(lat_tar_denorm, return_dict=False)[0]
        img_tar = self.pipe.image_processor.postprocess(img_tar)[0]

        # clean‑up GPU
        self.pipe = self.pipe.to("cpu")
        torch.cuda.empty_cache()

        return img_tar  # PIL.Image.Image


# ------------------------------------------------------------
# 3.  Batch runner (saves edited & diff, returns metrics)
# ------------------------------------------------------------
def run_edit(
    editor: FluxEditor,
    metric: MetricComputer,
    image_path: Path,
    source_prompt: str,
    target_prompt: str,
    *,
    negative_prompt: str | None,
    save_dir: Path,
    idx: int,
    num_steps: int,
    n_avg: int,
    source_guidance: float,
    target_guidance: float,
    n_min: int,
    n_max: int,
    seed: int,
) -> Dict[str, float]:
    init_pil = Image.open(image_path).convert("RGB")

    edited = editor.edit(
        init_pil,
        source_prompt,
        target_prompt,
        negative_prompt,
        num_steps,
        n_avg,
        source_guidance,
        target_guidance,
        n_min,
        n_max,
        seed,
    )

    # -------- save files --------
    stem, ext = image_path.stem, image_path.suffix or ".png"
    save_dir.mkdir(parents=True, exist_ok=True)
    edit_path = save_dir / f"{idx:04d}_{stem}_edited{ext}"
    diff_path = save_dir / f"{idx:04d}_{stem}_diff{ext}"
    edited.save(edit_path)
    ImageChops.difference(init_pil.resize(edited.size, Image.Resampling.LANCZOS), edited).save(diff_path)

    # -------- metrics ----------
    clip_edit = metric.clip_similarity(edited, target_prompt)
    clip_src  = metric.clip_similarity(init_pil, target_prompt)
    lpips_val = metric.lpips_distance(init_pil, edited)
    l1_mean, l1_med, l2_mean, l2_med = metric.pixel_distances(init_pil.resize(edited.size), edited)

    return {
        "clip_target_edit": clip_edit,
        "clip_target_src":  clip_src,
        "lpips":            lpips_val,
        "l1_mean":          l1_mean,
        "l1_median":        l1_med,
        "l2_mean":          l2_mean,
        "l2_median":        l2_med,
        "edited_path":      str(edit_path),
        "diff_path":        str(diff_path),
        "seed":             seed,
    }


# ------------------------------------------------------------
# 4.  CLI & main loop
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("FlowEdit Experiment Runner")
    p.add_argument("--data", type=str, default="dataset.json")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # FlowEdit hyper‑parameters (matching Gradio defaults)
    p.add_argument("--num_steps",        type=int,   default=28,  help="T_steps")
    p.add_argument("--n_avg",            type=int,   default=1)
    p.add_argument("--source_guidance",  type=float, default=1.5)
    p.add_argument("--target_guidance",  type=float, default=5.5)
    p.add_argument("--n_min",            type=int,   default=0)
    p.add_argument("--n_max",            type=int,   default=24)
    p.add_argument("--negative_prompt",  type=str,   default="")

    # misc
    p.add_argument("--save_dir", type=str, default="edited_flowedit")
    p.add_argument("--csv_out",  type=str, default="results_flowedit.csv")
    p.add_argument("--seed",     type=int, default=None, help="Global seed; leave blank for per‑sample randomness")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    editor = FluxEditor(args)
    metric = MetricComputer(device)

    with open(args.data) as f:
        dataset: List[Dict] = json.load(f)

    rows = []
    for idx, entry in enumerate(dataset):
        print(f"\n=== [{idx+1}/{len(dataset)}] {entry['image_path']} ===")
        seed = args.seed if args.seed is not None else int(torch.Generator(device="cpu").seed())

        stats = run_edit(
            editor,
            metric,
            Path(entry["image_path"]),
            entry["source_prompt"],
            entry["target_prompt"],
            negative_prompt=args.negative_prompt or None,
            save_dir=Path(args.save_dir),
            idx=idx,
            num_steps=args.num_steps,
            n_avg=args.n_avg,
            source_guidance=args.source_guidance,
            target_guidance=args.target_guidance,
            n_min=args.n_min,
            n_max=args.n_max,
            seed=seed,
        )
        for k, v in stats.items():
            print(f"{k:>20}: {v}")
        rows.append({**entry, **stats})

    # save CSV
    out = Path(args.csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFinished — detailed metrics stored in {out.resolve()}")


if __name__ == "__main__":
    main()
