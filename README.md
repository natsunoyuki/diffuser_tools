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

```
# text2img contains a class for the entire text to image pipeline.
# This class is able to handle clip skips, LoRAs and schedulers.
from diffuser_tools.text2img import Text2ImagePipe

# utilities contains functions for visualization and outupts.
from diffuser_tools.utilities import plot_images, save_images


# kMain safetensors directly from civitai.
model_dir = "meichidarkMix_meichidarkV5.safetensors"

# LoRA safetensors directly from civitai.
lora_path = "edgWar40KAdeptaSororitas.safetensors"

# Textual inversion.
textual_inversion_dirs = ["easynegative.safetensors", "badhandv4.pt"]
textual_inversion_tokens = ["easynegative", "badhandv4"]

# Clip skip.
clip_skip = 2

# Scheduler. 
scheduler = "DPMSMS"
scheduler_configs = {
    "use_karras_sigmas": True
}

# Create prompt and negative prompts.
prompt = """..."""
negative_prompt = """..."""


# Initialize the text to image pipe class.
text_2_img = Text2ImagePipe(
    model_dir = model_dir,
    prompt = prompt,
    negative_prompt = negative_prompt,
    lora_dir = lora_path,
    scheduler = scheduler,
    scheduler_configs = scheduler_configs,
    clip_skip = clip_skip,
    textual_inversion_dirs = textual_inversion_dirs,
    textual_inversion_tokens = textual_inversion_tokens,
    safety_checker = None,
    use_prompt_embeddings = True,
    use_compel = True,
    torch_dtype = torch_dtype,
    device = device_name
)


# Run the text to image pipeline for several seed values.
seeds = [i for i in range(0, 10)]
images = []
for seed in seeds:
    image = text_2_img.run_pipe(
        steps = 50,
        width = 512,
        height = 832,
        scale = 12.0,
        seed = seed,
        use_prompt_embeddings = True,
        verbose = False,
    )
    images.append(image)

plot_images(images)
```

## Converting CivitAI Safetensors for Diffusers
Many models released on <a href = "https://civitai.com">CivitAI</a> are published as safetensors. These safetensors must be converted to a format which the `diffusers` library can use. To do so, use a script provided on the `diffusers` official GitHub repository:
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
