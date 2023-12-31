# Diffuser Tools 
`diffuser_tools` a set of Python codes containing tools to make generating images using the <a href = "https://github.com/huggingface/diffusers">`diffusers`</a> package an easier task.

`diffuser_tools` contains functions to load and run `diffusers` `text2img` pipelines with prompt embeddings and clip-skipping, and visualize or save the generated images to disk. 

## Installation
```
pip install git+https://github.com/natsunoyuki/diffuser_tools
```

### Dependencies Versions
This set of Python codes has been tested with the following versions of the dependencies listed in `pyproject.toml`.
```
transformers==4.35.2
accelerate==0.25.0
diffusers==0.24.0
huggingface_hub==0.19.4
omegaconf==2.3.0
```

## Usage 
There are two main ways to use the `diffuser_tools` package.
1. Using the `Text2ImagePipe` class.
2. Using `diffuser_tools` functions.

The `Text2ImagePipe` class is the more recent method of using `diffuser_tools`, while the `diffuser_tools` functions are less recent and are kept for legacy purposese.

### ① Using the `Text2ImagePipe` Class
The `Text2ImagePipe` class contains the functionalities required to load pre-trained stable diffusion model safetensors, LoRA weight safetensors, use CLIP skip, and calculate prompt/negative prompt embeddings.

```
# text2img contains a class for the entire text to image pipeline.
# This class is able to handle clip skips, LoRAs and schedulers.
from diffuser_tools.text2img import Text2ImagePipe

# utilities contains functions for visualization and outupts.
from diffuser_tools.utilities import plot_images, save_images


# kMain safetensors directly from civitai.
model_dir = "kMain_kMain21.safetensors"

# LoRA safetensors directly from civitai.
lora_path = "edgWar40KAdeptaSororitas.safetensors"

# Clip skip.
clip_skip = 0

# Scheduler. 
scheduler = "EADS"

# Create prompt and negative prompts.
prompt = """..."""
negative_prompt = """..."""


# Initialize the text to image pipe class.
text_2_img = Text2ImagePipe(
    model_dir = model_dir,
    prompt = prompt,
    negative_prompt = negative_prompt,
    lora_path = lora_path,
    scheduler = scheduler,
    clip_skip = clip_skip,
    safety_checker = None,
    use_prompt_embeddings = True,
    split_character = ",",
    torch_dtype = torch_dtype,
    device = torch.device("cuda"),
)


# Run the text to image pipeline for several seed values.
seeds = [i for i in range(0, 5)]
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

### ② Using `diffuser_tools` Functions
The various `diffuser_tools` functions outlined below were created using older versions of diffusers which did not have many modern capabilities such as loading safetensors directly or loading LoRA weights.

Before CivitAI models can be used with the `diffuser_tools` functions outlined below, they need to be converted from safetensors to a format which older versions of the `diffusers` package can load. See the section <b>Converting CivitAI Safetensors for Diffusers</b> for more information on CivitAI safetensors conversion.

```
# utilities contains functions for visualization and outupts.
from diffuser_tools.utilities import plot_images, save_images

# prompt_utilities contains functions for creating prompt embeddings.
from diffuser_tools.prompt_utilities import get_prompt_embeddings

# text2img_utilities contain functions for creating and running 
# stable diffusion pipelines.
from diffuser_tools.text2img_utilities import load_pipeline, run_pipe


# Create stable diffusion text to image pipeline.
# Also, specify the last layer of the clip text encoder to use.
# clip_skip = 1 will use the full clip text encoder model
# (following standard conventions).
# clip_skip = 2 will use features from the second last layer
# (again following standard conventions).
pipe = load_pipeline(
    model_path,
    scheduler = "EADS",
    clip_skip = 2,
    clip_dir = clip_path,
    safety_checker = None,
    device_name = torch.device("cuda"),
    torch_dtype = torch.float16,
)

# Create prompt and negative prompts.
prompt = """..."""
negative_prompt = """..."""

# Create prompt embeddings.
prompt_embeds, negative_prompt_embeds = get_prompt_embeddings(
    pipe,
    prompt,
    negative_prompt,
    split_character = ",",
    device = torch.device("cuda"),
)

# Run stable diffusion pipeline to get images using clip_skip 
# and prompt embeddings.
images = run_pipe(
    pipe = pipe,
    prompt_embeddings = prompt_embeds,
    negative_prompt_embeddings = negative_prompt_embeds,
    steps = 50,
    width = 512,
    height = 768,
    scale = 7,
    seed = 123456789,
    n_images = 10,
    device_name = torch.device("cuda"),
)

# Visualize images.
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
