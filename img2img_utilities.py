import diffusers
import torch


# %% Load (image to image) diffuser pipeline.
def load_img2img_pipeline(
    model_dir,
    scheduler = None,
    device_name = torch.device("cpu"),
    torch_dtype = torch.float32
    ):
    pipe = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
        model_dir, torch_dtype = torch_dtype
    )

    if scheduler in ["EulerAncestralDiscreteScheduler", "EADS"]:
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
    # else default scheduler.

    pipe.safety_checker = lambda images, **kwargs: [images, [False] * len(images)]
    pipe = pipe.to(device_name)
    return pipe


# %% Run diffuser pipeline.
def run_pipe(pipe,
             prompt,
             image,
             strength = 0.8,
             num_inference_steps = 50,
             guidance_scale = 7.5,
             negative_prompt = None,
             n_images = 1,
             eta = 0,
             output_type = "pil",
             seed = 123456789,
             device_name = torch.device("cpu")):

    gen = torch.Generator(device = device_name).manual_seed(seed)

    image_list = []
    with torch.autocast("cuda"):
        for i in range(n_images):
            image = pipe(prompt = prompt,
                         image = image,
                         strength = strength,
                         num_inference_steps = num_inference_steps,
                         guidance_scale = guidance_scale,
                         negative_prompt = negative_prompt,
                         eta = eta,
                         output_type = output_type,
                         generator = gen)
            image_list = image_list + image.images

    return image_list