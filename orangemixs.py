import os
import shutil

import huggingface_hub


def get_orangemixs_files(repo_id = "WarriorMama777/OrangeMixs",
                         model_name = "AbyssOrangeMix2",
                         model_dir = None,
                         revision = "main"):
    """Downloads model files from WarriorMama777/OrangeMixs,
    preserving the directory structure
    for the diffusers library to use.

    There must be a simpler way to do this!
    """
    if model_dir is None:
        model_dir = model_name

    os.makedirs(model_dir, exist_ok = True)

    # Download model_index.json config file.
    path = huggingface_hub.hf_hub_download(repo_id = repo_id,
                                           filename = "model_index.json",
                                           revision = revision,
                                           cache_dir = "./")

    shutil.copy2(path, "{}/model_index.json".format(model_dir))

    # Download feature_extractor config file.
    part_name = "feature_extractor"
    filename = "Models/{}/Diffusers/{}/preprocessor_config.json".format(model_name, part_name)
    os.makedirs(os.path.join(model_dir, part_name), exist_ok = True)

    path = huggingface_hub.hf_hub_download(repo_id = repo_id,
                                           filename = filename,
                                           revision = revision,
                                           cache_dir = "./")
    shutil.copy2(path, "{}/{}/preprocessor_config.json".format(model_dir, part_name))

    # Download safety_checker files.
    part_name = "safety_checker"
    files = ["config.json", "pytorch_model.bin"]
    os.makedirs(os.path.join(model_dir, part_name), exist_ok = True)

    for f in files:
        filename = "Models/{}/Diffusers/{}/{}".format(model_name, part_name, f)
        path = huggingface_hub.hf_hub_download(repo_id = repo_id,
                                               filename = filename,
                                               revision = revision,
                                               cache_dir = "./")
        shutil.copy2(path, "{}/{}/{}".format(model_dir, part_name, f))

    # Download scheduler config file.
    part_name = "scheduler"
    filename = "Models/{}/Diffusers/{}/scheduler_config.json".format(model_name, part_name)
    os.makedirs(os.path.join(model_dir, part_name), exist_ok = True)

    path = huggingface_hub.hf_hub_download(repo_id = repo_id,
                                           filename = filename,
                                           revision = revision,
                                           cache_dir = "./")
    shutil.copy2(path, "{}/{}/scheduler_config.json".format(model_dir, part_name))

    # Download text_encoder files.
    part_name = "text_encoder"
    files = ["config.json", "pytorch_model.bin"]
    os.makedirs(os.path.join(model_dir, part_name), exist_ok = True)

    for f in files:
        filename = "Models/{}/Diffusers/{}/{}".format(model_name, part_name, f)
        path = huggingface_hub.hf_hub_download(repo_id = repo_id,
                                               filename = filename,
                                               revision = revision,
                                               cache_dir = "./")
        shutil.copy2(path, "{}/{}/{}".format(model_dir, part_name, f))

    # Download tokenizer files.
    part_name = "tokenizer"
    files = ["merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"]
    os.makedirs(os.path.join(model_dir, part_name), exist_ok = True)

    for f in files:
        filename = "Models/{}/Diffusers/{}/{}".format(model_name, part_name, f)
        path = huggingface_hub.hf_hub_download(repo_id = repo_id,
                                               filename = filename,
                                               revision = revision,
                                               cache_dir = "./")
        shutil.copy2(path, "{}/{}/{}".format(model_dir, part_name, f))

    # Download unet.
    part_name = "unet"
    files = ["config.json", "diffusion_pytorch_model.bin"]
    os.makedirs(os.path.join(model_dir, part_name), exist_ok = True)
    for f in files:
        filename = "Models/{}/Diffusers/{}/{}".format(model_name, part_name, f)
        path = huggingface_hub.hf_hub_download(repo_id = repo_id,
                                               filename = filename,
                                               revision = revision,
                                               cache_dir = "./")
        shutil.copy2(path, "{}/{}/{}".format(model_dir, part_name, f))

    # Download vae.
    part_name = "vae"
    os.makedirs(os.path.join(model_dir, part_name), exist_ok = True)
    files = ["config.json", "diffusion_pytorch_model.bin"]
    for f in files:
        filename = "Models/{}/Diffusers/{}/{}".format(model_name, part_name, f)
        path = huggingface_hub.hf_hub_download(repo_id = repo_id,
                                               filename = filename,
                                               revision = revision,
                                               cache_dir = "./")
        shutil.copy2(path, "{}/{}/{}".format(model_dir, part_name, f))

    return model_dir