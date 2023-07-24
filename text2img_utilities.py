import diffusers
import torch


# %% Load (text to image) diffuser pipeline.
def load_pipeline(
    model_dir,
    scheduler = None,
    safety_checker = False,
    device_name = torch.device("cpu"),
    torch_dtype = torch.float32
    ):
    """
    Loads a pre-trained diffusion pipeline downloaded from HuggingFace.

    Arguments
        model_dir: str
            Path to the downloaded model directory.
        scheduler: str or None
            Scheduler to use. Currently only "EDS", "EADS" or "DPMSMS"
            are supported. If None, default scheduler will be used.
        safety_checker: bool
            Turn on/off model safety checker.
        device_name: torch.device
            Device name to run the model on. Run on GPUs!
        torch_dtype: torch.float32 or torch.float16
            Dtype to run the model on. Choice of 32 bit or 16 bit floats.
            16 bit floats are less computationally intensive.
    Returns
        pipe: StableDiffusionPipeline
            Loaded diffuser pipeline.
    """
    # Load the pre-trained diffusion pipeline.
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype = torch_dtype
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
    if safety_checker is False:
        pipe.safety_checker = lambda images, **kwargs: [
            images, [False] * len(images)
        ]

    # Load the pipeline to the GPU if available.
    pipe = pipe.to(device_name)

    return pipe


# %% Run text2img diffuser pipeline.
def run_pipe(pipe,
             prompt,
             negative_prompt = None,
             steps = 50,
             width = 512,  # Multiple of 8
             height = 704, # Multiple of 8.
             scale = 8.0,
             seed = 123456789,
             n_images = 1,
             device_name = torch.device("cpu")):
    """
    Arguments
        pipe: StableDiffusionPipeline
            Stable diffusion pipeline from load_pipeline.
        prompt: str
            Prompt used to guide the denoising process.
        negative_prompt: str or None
            Negative prompt used to guide the denoising process.
            Used to restrict the possibilities of the output image.
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
        device_name: torch.device
            Device name to run the model on. Run on GPUs!
    Returns
        image_list: list
            List of output images.
    """
    if width % 8 != 0:
        print("Image width must be multiples of 8... adjusting!")
        width = int(width / 8) * 8
    if height % 8 != 0:
        print("Image width must be multiples of 8... adjusting!")
        height = int(height / 8) * 8

    # Using a torch.Generator allows for deterministic behaviour.
    gen = torch.Generator(device = device_name).manual_seed(seed)
    image_list = []

    with torch.autocast("cuda"):
        for i in range(n_images):
            image = pipe(prompt,
                         height = height,
                         width = width,
                         num_inference_steps = steps,
                         guidance_scale = scale,
                         negative_prompt = negative_prompt,
                         generator = gen)

            image_list = image_list + image.images

    return image_list
