import torch
import torch.nn.functional as F
import gradio as gr
from PIL import ImageChops, Image
import numpy as np
import random 

from diffusers import FluxPipeline
from FlowEdit_utils import FlowEditFLUX

class FluxEditor:
    def __init__(self):
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16).to("cuda")
    
    def edit(
        self,
        image,
        source_prompt,
        target_prompt,
        negative_prompt,
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
        diff = ImageChops.difference(
            init_resized.convert("RGB"),   # make sure both are RGB
            edited_image.convert("RGB")
        )
        self.pipe = self.pipe.to("cpu")
        torch.cuda.empty_cache()
        print("End Edit")
        
        arr1 = np.array(init_resized, dtype=np.float32)
        arr2 = np.array(edited_image, dtype=np.float32)

        # Compute L1 and L2 distances
        mae = np.mean(np.abs(arr1 - arr2))
        mae_median = np.median(np.abs(arr1 - arr2))
        mse = np.mean((arr1 - arr2) ** 2)
        mse_median = np.median((arr1 - arr2) ** 2)
        print("L1 Distance:", mae, "median:", mae_median)
        print("L2 Distance:", mse, "median:", mse_median)
        return edited_image, diff



def create_demo(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", offload: bool = False):
    editor = FluxEditor()

    with gr.Blocks() as demo:
        gr.Markdown(f"FlowEdit demo")
        
        with gr.Row():
            with gr.Column():
                source_prompt = gr.Textbox(label="Source Prompt", value="")
                target_prompt = gr.Textbox(label="Target Prompt", value="")
                source_guidance = gr.Slider(0.0, 10.0, 1.5, step=0.05, label="Source Guidance")
                target_guidance = gr.Slider(0.0, 10.0, 1.5, step=0.05, label="Target Guidance")
                init_image = gr.Image(label="Input Image", visible=True, type='pil')
                generate_btn = gr.Button("Generate")
                
                with gr.Accordion("Advanced Options", open=True):
                    negative_prompt = gr.Textbox(label="negative prompt for clip encoder", value=None)
                    num_steps = gr.Slider(1, 100, 28, step=1, label="Number of steps")
                    n_min = gr.Slider(1, 100, 0, step=1, label="min editing step")
                    n_max = gr.Slider(1, 100, 24, step=1, label="max editing step")
                    n_avg = gr.Slider(1, 100, 1, step=1, label="number of predictions to average")
                    seed = gr.Textbox(None, label="Seed")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                diff_image   = gr.Image(label="Difference (|input - output|)", type="pil")
        generate_btn.click(
            fn=editor.edit,
            inputs=[        
                init_image,
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
