import torch
import transformers
import diffusers


# Utility functions for text to image pipeline.


# %% Load (text to image) diffuser pipeline.
def load_pipeline(
    model_dir,
    scheduler = None,
    clip_skip = 1,
    clip_dir = None,
    safety_checker = None,
    device_name = torch.device("cpu"),
    torch_dtype = torch.float32,
    ):
    """
    Loads a pre-trained diffusion pipeline downloaded from HuggingFace.

    Arguments
        model_dir: str
            Path to the downloaded model directory.
        scheduler: str or None
            Scheduler to use. Currently only "EDS", "EADS" or "DPMSMS"
            are supported. If None, default scheduler will be used.
        clip_skip: int
            Number of clip layers to skip - 1 (as per community convention).
            If 1 all CLIP layers are used (no skipping).
            If 2 the last CLIP layer is skipped.
            Defaults to 1 (all CLIP layers used).
        clip_dir: str or None.
            Path to the CLIP text encoder model directory.
            If None model_dir will be used.
        safety_checker: None or SafetyChecker.
            Turn off safety checker with None.
        device_name: torch.device
            Device name to run the model on. Run on GPUs!
        torch_dtype: torch.float32 or torch.float16
            Dtype to run the model on. 
            Choice of 32 bit or 16 bit floats.
            16 bit floats are less computationally intensive.
            16 bit floats should be used for GPUs.
    Returns
        pipe: StableDiffusionPipeline
            Loaded diffuser pipeline.
    """
    # Load the pre-trained diffusion pipeline.
    # Account for clip skipping.    
    if clip_skip > 1:
        if clip_dir is None:
            clip_dir = model_dir

        text_encoder = transformers.CLIPTextModel.from_pretrained(
            clip_dir,
            subfolder = "text_encoder",
            num_hidden_layers = 12 - (clip_skip - 1),
            torch_dtype = torch_dtype
        )

        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype = torch_dtype,
            safety_checker = safety_checker,
            text_encoder = text_encoder,
        )
    else:
        pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype = torch_dtype,
            safety_checker = safety_checker,
        )


    # Change the scheduler.
    if scheduler in [
        "EulerAncestralDiscreteScheduler",
        "EADS"
    ]:
        pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    elif scheduler in ["EulerDiscreteScheduler", "EDS"]:
        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    elif scheduler in ["DPMSolverMultistepScheduler", "DPMSMS"]:
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
    # Else the default scheduler is used.

    # Change the safety checker.
    #if safety_checker is False:
    #    pipe.safety_checker = lambda images, **kwargs: [
    #        images, [False] * len(images)
    #    ]

    # Load the pipeline to the GPU if available.
    pipe = pipe.to(device_name)

    return pipe


# %% Run text2img diffuser pipeline.
def run_pipe(pipe,
             prompt = None,
             negative_prompt = None,
             prompt_embeddings = None,
             negative_prompt_embeddings = None,
             steps = 50,
             width = 512,  
             height = 768, 
             scale = 7.0,
             seed = 0,
             n_images = 1,
             device_name = None,
    ):
    """
    Arguments
        pipe: StableDiffusionPipeline
            Stable diffusion pipeline from load_pipeline.
        prompt: str or None.
            Prompt used to guide the denoising process.
        negative_prompt: str or None
            Negative prompt used to guide the denoising process.
            Used to restrict the possibilities of the output image.
        prompt_embedding: Tensor
            Prompt embeddings in lieu of prompts.
        negative_prompt_embedding: Tensor
            Negative prompt embeddings in lieu of negative prompts.
        steps: int
            Number of denoising iterations.
        width, height: int
            Dimensions of the output image. Must be a multiple of 8.
        scale: float
            Scale which controls how much the model follows the prompt.
            Higher values lead to more imaginative outputs.
        seed: int
            Random seed used to initialize the noisy image.
        n_images: int
            How many images to produce.
            If n_images > 1, seed will be iteratively increased to
            generate the new images.
        device_name: torch.device
            Device name to run the model on. Run on GPUs!
    Returns
        image_list: list
            List of output images.
    """
    if device_name is not None:
        pipe = pipe.to(device_name)

    # Multiple seeds.
    seeds = [i for i in range(seed, seed + n_images, 1)]

    images = []

    if prompt is None and prompt_embeddings is None:
        return images
    
    for seed in seeds:
        if prompt is not None:
            new_img = pipe(
                prompt = prompt,
                negative_prompt = negative_prompt,
                width = width,
                height = height,
                guidance_scale = scale,
                num_inference_steps = steps,
                num_images_per_prompt = 1,
                generator = torch.manual_seed(seed),
            ).images
        else:
            new_img = pipe(
                prompt_embeds = prompt_embeddings,
                negative_prompt_embeds = negative_prompt_embeddings,
                width = width,
                height = height,
                guidance_scale = scale,
                num_inference_steps = steps,
                num_images_per_prompt = 1,
                generator = torch.manual_seed(seed),
            ).images

        images = images + new_img

    return images