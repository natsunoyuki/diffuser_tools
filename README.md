# Diffuser Tools 
`diffuser_tools` a set of Python codes containing tools to make generating images using the <a href = "https://github.com/huggingface/diffusers">`diffusers`</a> package an easier task.

`diffuser_tools` contains functions to load and run `diffusers` `text2img` pipelines with prompt embeddings and clip-skipping, and visualize or save the generated images to disk. 

## Installation
```
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

## Usage 
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

# Model safetensors from civitai.
model_path = "kMain_kMain21.safetensors"

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
scheduler_configs = {
    "use_karras_sigmas": True
}

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
    img2img = False,
    torch_dtype = torch_dtype,
    device = device_name
)


# Run the text to image pipeline for several seed values.
start_seed = 0
N_imgs = 10
seeds = [i for i in range(start_seed, start_seed + N_imgs)]
images = []
for seed in seeds:
    image = text_2_img.run_pipe(
        steps = 30,
        width = 512,
        height = 832,
        scale = 7,
        seed = seed,
        use_prompt_embeddings = True,
        verbose = False
    )
    images.append(image)
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
