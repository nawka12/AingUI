# AingUI
A lightweight and simple Stable Diffusion WebUI for SDXL optimized for Animagine XL 3.x and its derivatives

AingUI is a simple and lightweight Stable Diffusion UI built with Gradio on top of Diffusers. Designed to work with SDXL and optimized for Animagine XL 3.x and its derivatives.

## Installation
(Tested with Python 3.11.5)
### Windows
```pwsh
# Use Powershell to run this.
git clone https://github.com/nawka12/AingUI # Clone this repository
cd AingUI
python -m venv venv # Create virtual environment
venv\Scripts\activate # Activate virtual environment
pip install -r requirements.txt # Install requirements

py app.py --model-path MODEL_PATH_HERE
```

### Linux
```bash
git clone https://github.com/nawka12/AingUI # Clone this repository
cd AingUI
python -m venv venv # Create virtual environment
venv/bin/activate # Activate virtual environment
pip install -r requirements.txt # Install requirements

py app.py --model-path MODEL_PATH_HERE
```
## Features
- **Built on top of Diffusers**: AingUI is built on top of Diffusers. Making it compatible with the Diffusers and Safetensors model format.

- **Directly pull a model from HuggingFace**: You can directly pull a model from HuggingFace by providing the link to the `.safetensors` file or directory of a Diffusers model.

- **Offline Model**: You can use a model saved on your device by providing an offline directory of a model.

- **Single Sampler**: No need to worry about what sampler to use or even "what are samplers?". We use the most optimized sampler for Animagine XL 3.x, `Euler a`, by default.

- **Longer Prompt Support**: AingUI implements `compel` library to support prompts longer than 77 tokens, which is not supported by Diffusers by default.

- **Metadata**: AingUI saves and uses the same metadata format used in AUTO111/Forge. So you can share your generated images to [Civitai](https://civitai.com) or other platforms with all your parameters readable.

- **Resolution**: No need to worry that the resolution you set will produce bad images. Our resolution radio button lists all the resolutions supported by SDXL.

- **Quality Prompts**: AingUI includes a checkbox to automatically add the standard quality prompts recommended for Animagine XL 3.x to your custom prompt.

## App Parameters
- `--model-path`: Path to the model file/diffusers directory. Required.
- `--auth [user:pass]`: Add login page to AingUI. Use `username:password` format to set the credential.
- `--listen`: Listen on all network interfaces. Used for port forwarding.
- `--port`: Set custom port (default: 7860)
