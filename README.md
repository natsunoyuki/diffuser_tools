# Diffuser Tools 
`diffuser_tools` a set of Python codes containing tools to make generating images using the <a href = "https://github.com/huggingface/diffusers">`diffusers`</a> package an easier task.

`diffuser_tools` contains functions to load and run `diffusers` `text2img` pipelines with prompt embeddings and clip-skipping, and visualize or save the generated images to disk. 

## Installation
Clone this repository, and install locally.
```bash
git clone https://github.com/natsunoyuki/diffuser_tools.git
cd diffuser_tools
pip install .
```

Alternatively, perform a direct pip install from the repository.
```bash
pip install git+https://github.com/natsunoyuki/diffuser_tools
```

### Dependencies Versions
This set of Python codes has been tested on 2024 July 15 on Google Colab with the following dependencies:
```
transformers==4.41.2
accelerate==0.32.1
diffusers==0.29.2
huggingface_hub==0.23.4
omegaconf==2.3.0
compel==2.0.2
peft==0.11.1
```

## API Usage 
The `Text2ImagePipe` class contains the pipeline for loading and running stable diffusion models with the `diffusers` package.
 method of using `diffuser_tools`, while the `diffuser_tools` functions are less recent and are kept for legacy purposese.

The `Text2ImagePipe` class contains the functionalities required to load pre-trained stable diffusion model safetensors, LoRA weight safetensors, use CLIP skip, and calculate prompt/negative prompt embeddings.

```python
# text2img contains a class for the entire text to image pipeline.
# This class is able to handle clip skips, LoRAs and schedulers.
from diffuser_tools.text2img import Text2ImagePipe

# utilities contains functions for visualization and outupts.
from diffuser_tools.utilities import plot_images, save_images

# Set GPU or CPU.
import torch

if torch.backends.mps.is_available():
    device_name = torch.device("mps")
    torch_dtype = torch.float32
elif torch.cuda.is_available():
    device_name = torch.device("cuda")
    torch_dtype = torch.float16
else:
    device_name = torch.device("cpu")
    torch_dtype = torch.float32

# Prompts.
prompt = "..."
negative_prompt = "..."

# Diffusion pipeline model converted from civitai safetensors.
# See the section "Converting CivitAI Safetensors for Diffusers" below.
# Now that runwayml has removed stable diffusion 1.5 weights from 
# HuggingFace, using the downloaded .safetensors from civitai does
# not work, as the pipeline will still attempt to download certain files
# from HuggingFace. 
model_path = "kMain/"

# Textual inversion safetensors and pytorch files from civitai.
textual_inversion_paths = ["easynegative.safetensors", "badhandv4.pt"]
textual_inversion_tokens = ["easynegative", "badhandv4"]

# Lora safetensors from civitai.
lora_paths = ["tangbohu-line_1.0.safetensors"]
lora_adapter_names = ["tangbohu-line"]
lora_scales = [0.7]

# CLIP skip.
clip_skip = 2

# Scheduler.
scheduler = "DPMSMS"
scheduler_configs = {"use_karras_sigmas": True}

text_2_img = Text2ImagePipe(
    model_path = model_path,
    prompt = prompt,
    negative_prompt = negative_prompt,
    lora_paths = lora_paths,
    lora_adapter_names = lora_adapter_names,
    lora_scales = lora_scales,
    scheduler = scheduler,
    scheduler_configs = scheduler_configs,
    clip_skip = clip_skip,
    textual_inversion_paths = textual_inversion_paths,
    textual_inversion_tokens = textual_inversion_tokens,
    safety_checker = None,
    use_prompt_embeddings = True,
    use_compel = True,
    img2img = False, # For img2img pipeline.
    torch_dtype = torch_dtype,
    device = device_name
)

# Run the image generation pipeline.
N_imgs = 10
width = 512
height = 768
seed = 4
steps = 30
scale = 5
images = text_2_img.run_pipe(
    steps = steps,
    width = width,
    height = height,
    scale = scale,
    seed = seed,
    #image = [seed_image] * N_imgs, # For img2img pipeline.
    #strength = 0.8, # For img2img pipeline.
    use_prompt_embeddings = True,
    verbose = False,
    num_images_per_prompt = N_imgs,
)
```

## Command Line Usage
Ensure that all the model weights are downloaded and converted to the `diffusers` format (see "Converting CivitAI Safetensors for Diffusers" below). Change any configurations in `main.yml` as required, and run `main.py` from the command line.
```bash
python3 main.py
```
Alternatively, create new configuration files in some sub-directory of `diffuser_tools/` and run `main.py` from the command line while specifying the sub-directory and file name.
```bash
python3 main.py --config_dir example_configs --config main_config.yaml
```

## Converting CivitAI Safetensors for Diffusers
Many models released on <a href = "https://civitai.com">CivitAI</a> are published as safetensors. For older versions of `diffusers<=0.24.0`, these safetensors might need to be converted to a format which the older `diffusers<=0.24.0` library can use. To do so, use a script provided on the `diffusers<=0.24.0` official GitHub repository:
```
wget https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_original_stable_diffusion_to_diffusers.py
```
where `main` should follow the version of `diffusers` used. Therefore if `diffusers==0.24.0` then the following version should be used instead:
```
wget https://raw.githubusercontent.com/huggingface/diffusers/v0.24.0/scripts/convert_original_stable_diffusion_to_diffusers.py
```
The safetensor files downloaded from <a href = "https://civitai.com">CivitAI</a> can then be converted using the command line:
```
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path <model_name>.safetensors --dump_path <output_path>/ --from_safetensors
```
