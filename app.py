import os
import torch
import random
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerAncestralDiscreteScheduler
import gradio as gr
from datetime import datetime
from PIL import PngImagePlugin
import argparse
from compel import Compel, ReturnedEmbeddingsType

parser = argparse.ArgumentParser(description='AingUI Image Generation')
parser.add_argument('--port', type=int, default=7860, help='Server port (default: 7860)')
parser.add_argument('--listen', action='store_true', help='Listen on all network interfaces (default: False)')
parser.add_argument('--auth', nargs='?', const='username:password', help='Set username and password for Gradio app (default: username:password)')
parser.add_argument('--model-path', type=str, required=True, help='Path to the model file (required)')
args = parser.parse_args()

if args.auth:
    username, password = args.auth.split(':')
else:
    username, password = None, None

model_path = args.model_path

if os.path.isdir(model_path):
    model_name = os.path.basename(model_path)  # Extract the directory name
    print(f"Loading {model_name} Diffusers model...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.to('cuda')

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
elif model_path.endswith('.safetensors'):
    model_name = os.path.basename(model_path).split('.safetensors')[0]  # Extract the filename without extension
    print(f"Loading {model_name} Safetensors model...")

    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.to('cuda')

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
else:
    print("Invalid input. Please provide either a directory or a .safetensors file.")
    exit()  # Exit the script if the input is invalid

def generate_image(prompt, negative_prompt, use_seed_randomizer, custom_seed, enable_standard_quality, num_inference_steps, guidance_scale, resolution):
    if use_seed_randomizer:
        seed = random.randint(1, 999999999999)  # Maximum seed number is 12 digits
        torch.manual_seed(seed)
        random.seed(seed)
        custom_seed = f"{seed}"
    else:
        if custom_seed:
            seed = int(custom_seed)
            torch.manual_seed(seed)
            random.seed(seed)
            custom_seed = f"{seed}"
        else:
            custom_seed = "Seed not specified."
    
    # Update prompt based on checkbox state
    if enable_standard_quality:
        prompt += ", masterpiece, best quality, very aesthetic, absurdres,"
        negative_prompt = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, watermark, artistic error, username, scan, [abstract]," + negative_prompt

    compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , 
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2], 
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
                requires_pooled=[False, True],
               truncate_long_prompts=False)
    
    conditioning, pooled = compel([prompt, negative_prompt])

    width, height = resolution.split(" x ")
    width = int(width)
    height = int(height)

    output = pipe(
        prompt_embeds=conditioning[0:1], pooled_prompt_embeds=pooled[0:1], 
        negative_prompt_embeds=conditioning[1:2], negative_pooled_prompt_embeds=pooled[1:2],
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )

    image = output.images[0]
    image_width, image_height = image.size

    # Create directory for saving images
    current_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = f"generation/{current_date}"
    os.makedirs(save_dir, exist_ok=True)

    # Create metadata dictionary
    metadata = PngImagePlugin.PngInfo()
    metadata_text = f"{prompt}\nNegative prompt: {negative_prompt}\nSteps: {num_inference_steps}, Size: {image_width}x{image_height}, Seed: {custom_seed}, Model: {model_name}, Version: AingUI, Sampler: Euler a, CFG scale: {guidance_scale},"
    metadata.add_text("parameters", metadata_text)

    # Save the generated image with metadata
    save_path = os.path.join(save_dir, f"generated_image_{datetime.now().strftime('%H%M%S')}.png")
    image.save(save_path, pnginfo=metadata)

    # Free GPU memory after generation
    torch.cuda.empty_cache()

    return save_path, custom_seed, metadata_text

css_style = """
img {
    max-height: 70vh;
    width: auto;
    display: block;
    margin: 0 auto;
}
label.prompt {
    font-weight: bold;
    margin-top: 10px;
}
h1 {
    text-align: center;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
}

/* Styles for smaller screens */
@media (max-width: 768px) {
    .prompt-group {
        flex-direction: column;
    }
}
"""

block = gr.Blocks(title="AingUI", css=css_style, theme="NoCrypt/miku@1.2.1")

with block:
    gr.HTML("<h1>AingUI</h1>")
    gr.HTML(f"<h2><center>Running {model_name}</center></h2>")
    with gr.Row(elem_classes="prompt-group"):
        prompt_input = gr.TextArea(label="Prompt", placeholder="Enter your prompt here")
        negative_prompt_input = gr.TextArea(label="Negative Prompt", placeholder="Enter your negative prompt here")
    num_inference_steps_input = gr.Slider(minimum=1, maximum=50, value=28, step=1, label="Number of Inference Steps")
    guidance_scale_input = gr.Slider(minimum=1, maximum=12, value=7, step=0.5, label="Guidance Scale")
    resolution_input = gr.Radio(["1024 x 1024", "1152 x 896", "896 x 1152", "1216 x 832", "832 x 1216", "1344 x 768", "768 x 1344", "1536 x 640", "640 x 1536"], value="896 x 1152", label="Resolution")
    custom_seed_input = gr.Textbox(label="Seed", placeholder="Enter custom seed (if not using randomizer)", max_lines=1)
    use_seed_randomizer_input = gr.Checkbox(label="Use Seed Randomizer", value=True)
    enable_standard_quality_input = gr.Checkbox(label="Enable Standard Quality Prompt for Animagine XL and Its Derivatives", value=True)
    generate_button = gr.Button("Generate Image")
    output_image = gr.Image(label="Generated Image")
    prompt_label = gr.TextArea(label="Final Prompt", interactive=False)
    generate_button.click(fn=generate_image, inputs=[prompt_input, negative_prompt_input, use_seed_randomizer_input, custom_seed_input, enable_standard_quality_input, num_inference_steps_input, guidance_scale_input, resolution_input], outputs=[output_image, custom_seed_input, prompt_label])
    custom_seed_input.disabled = True

server_port = args.port
server_address = '0.0.0.0' if args.listen else '127.0.0.1'

if username and password:
    block.launch(server_name=server_address, server_port=server_port, auth=(username, password))
else:
    block.launch(server_name=server_address, server_port=server_port)
