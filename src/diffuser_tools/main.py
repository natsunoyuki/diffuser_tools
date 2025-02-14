import pathlib
import torch
from PIL import Image

from diffuser_tools.text2img import Text2ImagePipe


# Command line main function.
def main(**configs):
    if torch.backends.mps.is_available():
        device_name = torch.device("mps")
        torch_dtype = torch.float32
    elif torch.cuda.is_available():
        device_name = torch.device("cuda")
        torch_dtype = torch.float16
    else:
        device_name = torch.device("cpu")
        torch_dtype = torch.float32
    print("Using {} on {}.".format(torch_dtype, device_name))

    model_dir = configs.get("model_dir", None)
    if model_dir is None:
        print("model_dir not specified. Terminating.")
        return
    else:
        print("Loading model from `{}`.".format(model_dir))
    
    textual_inversion_dirs = configs.get("textual_inversion_dirs", [])
    textual_inversion_tokens = configs.get("textual_inversion_tokens", [])
    for tid, tii in zip(textual_inversion_dirs, textual_inversion_tokens):
        print("Textual inversion: {}, {}.".format(tid, tii))

    loras = configs.get("loras", [[]])
    lora_paths, lora_adapter_names, lora_scales = [], [], []
    for l in loras:
        lora_paths.append(l[0])
        lora_adapter_names.append(l[1])
        lora_scales.append(l[2])
        print("Lora: {}.".format(l))

    clip_skip = configs.get("clip_skip", 0)
    print("Clip skip: {}.".format(clip_skip))

    scheduler = configs.get("scheduler", "DPMSMS")
    scheduler_configs = configs.get("scheduler_configs", {"use_karras_sigmas": True})
    print("Scheduler: {}, {}.".format(scheduler, scheduler_configs))

    prompt = configs.get("prompt", None)
    if prompt is None:
        return
    else:
        prompt = "".join(str(prompt).split("\n"))
    
    negative_prompt = configs.get("negative_prompt", None)
    if negative_prompt is not None:
        negative_prompt = "".join(str(negative_prompt).split("\n"))

    safety_checker = configs.get("safety_checker", None)
    use_prompt_embeddings = configs.get("use_prompt_embeddings", True)
    use_compel = configs.get("use_compel", True)
    img2img_image_path = configs.get("img2img_image_path", None)
    if img2img_image_path is not None:
        img2img = True
    else:
        img2img = False

    text_2_img = Text2ImagePipe(
        model_path = model_dir,
        prompt = prompt,
        negative_prompt = negative_prompt,
        lora_paths = lora_paths,
        lora_adapter_names = lora_adapter_names,
        lora_scales = lora_scales,
        scheduler = scheduler,
        scheduler_configs = scheduler_configs,
        clip_skip = clip_skip,
        textual_inversion_paths = textual_inversion_dirs,
        textual_inversion_tokens = textual_inversion_tokens,
        safety_checker = safety_checker,
        use_prompt_embeddings = use_prompt_embeddings,
        use_compel = use_compel,
        img2img = img2img,
        torch_dtype = torch_dtype,
        device = device_name
    )

    num_images_per_prompt = configs.get("num_images_per_prompt", 10)
    width = configs.get("width", 512)
    height = configs.get("height", 768)
    seed = configs.get("seed", 0)
    steps = configs.get("steps", 30)
    scale = configs.get("scale", 7)
    strength = configs.get("strength", 0.8)

    if img2img_image_path is not None:
        try:
            seed_image = Image.open(img2img_image_path)
            iW, iH = seed_image.size
            if iW != width and iH != height:
                scale = min(iH / height, iW / width)
                seed_image = seed_image.resize([int(iW / scale), int(iH / scale)])
            print("Input image {} dimensions: {}.".format(img2img_image_path, seed_image.size))
        except:
            seed_image = None
    else:
        seed_image = None

    # Image generation.
    images = text_2_img.run_pipe(
        steps = steps,
        width = width,
        height = height,
        scale = scale,
        seed = seed,
        image = [seed_image] * num_images_per_prompt,
        strength = strength,
        use_prompt_embeddings = use_prompt_embeddings,
        verbose = False,
        num_images_per_prompt = num_images_per_prompt,
    )

    # Output to disk.
    output_dir = pathlib.Path(configs.get("output_dir", "outputs"))
    if output_dir.exists() is False:
        output_dir.mkdir(exist_ok = True)

    project_name = configs.get("project_name", None)
    if project_name is None:
        project_name = str(model_dir).split("/")[-1]
        
    for count, im in enumerate(images):
        out_im_path = output_dir / "{}_{}.png".format(project_name, count+1)
        print("Saving to: {}.".format(out_im_path))
        im.save(out_im_path)
