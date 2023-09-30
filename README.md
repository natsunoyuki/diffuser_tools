# Diffuser Tools 
`diffuser_tools` a set of Python codes containing tools to make generating images using the <a href = "https://github.com/huggingface/diffusers">`diffusers`</a> package an easier task.

`diffuser_tools` contains functions to load and run `diffusers` `text2img` pipelines with prompt embeddings and clip-skipping, and visualize or save the generated images to disk. 

## Installation
```
pip install git+https://github.com/natsunoyuki/diffuser_tools
```

### Versions of Dependencies
This set of Python codes has been tested with the following versions of the dependencies listed in `pyproject.toml`.
```
transformers==4.31.0
accelerate==0.21.0
diffusers==0.20.0
huggingface_hub==0.16.4
omegaconf==2.3.0
```

## Usage
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

### Using CivitAI Models
Many models released on <a href = "https://civitai.com">CivitAI</a> are published as safetensors. These safetensors must be converted to a format which the `diffusers` library can use. To do so, use a script provided on the `diffusers` official GitHub repository:
```
wget https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_original_stable_diffusion_to_diffusers.py
```
where `main` should follow the version of `diffusers` used. Therefore if `diffusers==0.20.0` then the following version should be used instead:
```
wget https://raw.githubusercontent.com/huggingface/diffusers/v0.20.0/scripts/convert_original_stable_diffusion_to_diffusers.py
```
The safetensor files downloaded from <a href = "https://civitai.com">CivitAI</a> can then be converted using the command line:
```
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path kMain_kMain21.safetensors --dump_path kMain/ --from_safetensors
```
