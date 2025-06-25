# Copyright 2025 Natsunoyuki.

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

import torch


# Utility functions for prompts and negative prompts.
# %% Prompt embeddings to overcome CLIP 77 token limit.
# https://github.com/huggingface/diffusers/issues/2136
def get_prompt_embeddings(
    pipe,
    prompt,
    negative_prompt,
    device = torch.device("cpu")
):
    """Prompt embeddings to overcome CLIP 77 token limit.
    https://github.com/huggingface/diffusers/issues/2136
    """
    max_length = pipe.tokenizer.model_max_length

    input_ids = pipe.tokenizer(prompt, return_tensors = "pt", truncation = False).input_ids.to(device)
    negative_ids = pipe.tokenizer(negative_prompt, return_tensors = "pt", truncation = False).input_ids.to(device)

    if input_ids.shape[-1] >= negative_ids.shape[-1]:
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipe.tokenizer(
            negative_prompt, return_tensors = "pt", truncation = False, 
            padding = "max_length", max_length = shape_max_length
        ).input_ids.to(device)
    else:
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipe.tokenizer(
            prompt, return_tensors = "pt", truncation = False, 
            padding = "max_length", max_length = shape_max_length
        ).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

    prompt_embeddings = torch.cat(concat_embeds, dim = 1)
    negative_prompt_embeddings = torch.cat(neg_embeds, dim = 1)

    return prompt_embeddings, negative_prompt_embeddings