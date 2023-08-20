# Diffuser Tools 
This is a set of Python codes containing tools for generating images using stable diffusion models using the `diffusers` package.

## Installation
```
pip install git+https://github.com/natsunoyuki/diffuser_tools
```

## Usage
```
# utilities contains functions for visualization and outupts.
from diffuser_tools.utilities import plot_images, save_images

# prompt_utilities contains functions for creating prompt embeddings.
from diffuser_tools.prompt_utilities import get_prompt_embeddings

# text2img_utilities contain functions for creating and running stable diffusion pipelines.
from diffuser_tools.text2img_utilities import load_pipeline, run_pipe


# Create stable diffusion pipeline.
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

# Run stable diffusion pipeline to get images.
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