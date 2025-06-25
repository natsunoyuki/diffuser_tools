# Copyright 2025 Natsunoyuki AI Laboratory

# diffuser_tools is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) any 
# later version.

# diffuser_tools is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.

# You should have received a copy of the GNU General Public License along with 
# diffuser_tools. If not, see <https://www.gnu.org/licenses/>.

import os
import shutil
import huggingface_hub


# Functions for downloading orange mix model files from HuggingFace.
# https://huggingface.co/WarriorMama777/OrangeMixs/tree/main/Models/AbyssOrangeMix2/Diffusers


# %% Download orange mix files from HuggingFace.
# TODO improve and clean the code up.
def get_orangemixs_files(repo_id = "WarriorMama777/OrangeMixs",
                         model_name = "AbyssOrangeMix2",
                         model_dir = None,
                         revision = "main"):
    """Downloads model files from WarriorMama777/OrangeMixs,
    preserving the directory structure
    for the diffusers library to use.

    There must be a simpler way to do this!

    Arguments:
        repo_id: str 
            Path to the HuggingFace model repository.
        model_name: str
            Model name on the HuggingFace model repository.
        model_dir: None or str
            Local directory name where the model files will be downloaded to.
        revision: str
            HuggingFace model repository version. Defaults to "main".
    Returns:
        model_dir: None or str
            Local directory name where the model files will be downloaded to.
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