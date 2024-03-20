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