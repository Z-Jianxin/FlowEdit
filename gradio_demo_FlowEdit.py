import torch
import torch.nn.functional as F
import gradio as gr
from PIL import ImageChops, Image
import numpy as np
import random 

from diffusers import FluxPipeline
from FlowEdit_utils import FlowEditFLUX

from transformers import CLIPProcessor, CLIPModel
import lpips
import torchvision.transforms as transforms

import timm

import csv
import os


csv_file = "/lambda/nfs/DISK0/experiments/logs.tsv"

# If file doesn't exist yet, write the header
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["seed", "target image clip", "source image clip", "L1 distance", "L2 distance", "LPIPS", "DINO"])


class FluxEditor:
    def __init__(self):
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16).to("cuda")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336") # it's lightweighted so it can live on CPU
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.lpips = lpips.LPIPS(net='vgg').to('cuda')
        self.lpips_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # resize for consistency (optional)
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # LPIPS expects inputs in [-1, 1]
        ])
        
                
        self.dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=0).to('cuda')
        self.dino.eval()
        self.dino_transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def dino_dist(self, img1, img2):
        img1 = self.dino_transform(img1).unsqueeze(0)
        img2 = self.dino_transform(img2).unsqueeze(0)
        with torch.no_grad():
            emb1 = F.normalize(self.dino(img1.to('cuda')), dim=1)
            emb2 = F.normalize(self.dino(img2.to('cuda')), dim=1)
        return F.cosine_similarity(emb1, emb2).item()

    def print_clip_score(self, image, prompt):
        clip_inputs = self.clip_processor(
            text=[prompt,],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        clip_outputs = self.clip_model(**clip_inputs)
        clip_score = clip_outputs.logits_per_image.detach().item()
        print("CLIP score: ", clip_score)
        return clip_score

    def edit(
        self,
        image,
        source_prompt,
        target_prompt,
        negative_prompt,
        eval_prompt,
        T_steps,
        n_avg,
        source_guidance_scale,
        target_guidance_scale,
        n_min,
        n_max,
        seed,
    ):
        if not seed:
            seed = torch.Generator(device="cpu").seed()
        print("Random seed: ", seed)
        #random.seed(seed)
        #np.random.seed(seed)
        torch.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)

        device = torch.device("cuda")
        self.pipe = self.pipe.to(device)

        negative_prompt = negative_prompt if negative_prompt else None

        image_src = self.pipe.image_processor.preprocess(image, height=1024, width=1024)
        # cast image to half precision
        image_src = image_src.to(device).half()
        with torch.autocast("cuda"), torch.inference_mode():
            x0_src_denorm = self.pipe.vae.encode(image_src).latent_dist.mode()
        x0_src = (x0_src_denorm - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        # send to cuda
        x0_src = x0_src.to(device)
            
        x0_tar = FlowEditFLUX(
            self.pipe,
            self.pipe.scheduler,
            x0_src,
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


        x0_tar_denorm = (x0_tar / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            image_tar = self.pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
        image_tar = self.pipe.image_processor.postprocess(image_tar)[0]

        edited_image = image_tar
        init_resized = image.resize(
            edited_image.size,              # match W×H
            Image.Resampling.LANCZOS        # high-quality down/up-sampling
        )
        diff_img = ImageChops.difference(
            init_resized.convert("RGB"),   # make sure both are RGB
            edited_image.convert("RGB")
        )
        self.pipe = self.pipe.to("cpu")

        print("\ntarget prompt vs target image: ")
        target_clip = self.print_clip_score(edited_image, target_prompt)
        print("target prompt vs source image: ")
        source_clip = self.print_clip_score(init_resized, target_prompt)
        print("source prompt vs target image: ")
        self.print_clip_score(edited_image, source_prompt)
        print("source prompt vs source image: ")
        self.print_clip_score(init_resized, source_prompt)

        a1 = np.asarray(init_resized, dtype=np.float32)
        a2 = np.asarray(edited_image, dtype=np.float32)
        # per‑channel difference
        diff_np = a1 - a2
        # average across the 3 channels so l1/l2 match image (H, W)
        l1 = np.mean(np.abs(diff_np), axis=-1)      # shape (H, W)
        l2 = np.mean(diff_np ** 2,  axis=-1)        # shape (H, W)
        l1_mean = l1.mean()
        l2_mean = l2.mean()

        print("L1 Distance:", l1_mean, "median:", np.median(l1))
        print("L2 Distance:", l2_mean, "median:", np.median(l2))
        
        with torch.no_grad():
            lpips_dist = self.lpips(self.lpips_transform(init_resized).to("cuda"), self.lpips_transform(edited_image).to("cuda")).item()
        print("LPIPS distance: ", lpips_dist)
        
        with torch.no_grad():
            dino_sim = self.dino_dist(init_resized, edited_image)
            dino_dist = 1 - dino_sim
        print("DINO distance: ", dino_dist)

        torch.cuda.empty_cache()
        print("End Edit\n\n")
        
        # Append the new row
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([
                int(seed),
                f"{target_clip:.8f}",
                f"{source_clip:.8f}",
                f"{l1_mean:.8f}",
                f"{l2_mean:.8f}",
                f"{lpips_dist:.8f}",
                f"{dino_dist:.8f}"
            ])
        return edited_image, diff_img



def create_demo(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", offload: bool = False):
    editor = FluxEditor()

    with gr.Blocks() as demo:
        gr.Markdown(f"FlowEdit demo")
        
        with gr.Row():
            with gr.Column():
                source_prompt = gr.Textbox(label="Source Prompt", value="")
                target_prompt = gr.Textbox(label="Target Prompt", value="")
                source_guidance = gr.Slider(0.0, 10.0, 1.5, step=0.05, label="Source Guidance")
                target_guidance = gr.Slider(0.0, 10.0, 5.5, step=0.05, label="Target Guidance")
                generate_btn = gr.Button("Generate")
                
                with gr.Accordion("Advanced Options", open=True):
                    eval_prompt = gr.Textbox(label="Prompt to evaluate generative quality", value=None)
                    negative_prompt = gr.Textbox(label="negative prompt for clip encoder", value=None)
                    num_steps = gr.Slider(1, 100, 28, step=1, label="Number of steps")
                    n_min = gr.Slider(1, 100, 0, step=1, label="min editing step")
                    n_max = gr.Slider(1, 100, 24, step=1, label="max editing step")
                    n_avg = gr.Slider(1, 100, 1, step=1, label="number of predictions to average")
                    seed = gr.Textbox(None, label="Seed")
            with gr.Column():
                init_image = gr.Image(label="Input Image", visible=True, type='pil')

            with gr.Column():
                output_image = gr.Image(label="Generated Image", format='jpg')

            with gr.Column():
                diff_image   = gr.Image(label="Difference (|input - output|)", format='jpg')

        generate_btn.click(
            fn=editor.edit,
            inputs=[        
                init_image,
                source_prompt,
                target_prompt,
                negative_prompt,
                eval_prompt,
                num_steps,
                n_avg,
                source_guidance,
                target_guidance,
                n_min,
                n_max,
                seed,
            ],
            outputs=[output_image, diff_image]
        )


    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")

    parser.add_argument("--port", type=int, default=43035)
    args = parser.parse_args()

    demo = create_demo("SDE coupling Demo with Flux", args.device, args.offload)
    demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)
