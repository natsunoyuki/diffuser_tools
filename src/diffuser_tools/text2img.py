import sys
import time

import diffusers
import torch

# TODO

class Text2ImagePipe(object):
    # %%
    def __init__(
        self, 
        model_dir, 
        prompt = None, 
        negative_prompt= None,
        lora_dir = None,
        scheduler = None,
        clip_skip = 1,
        safety_checker = None,
        use_prompt_embeddings = True,
        split_character = ",",
        torch_dtype = torch.float32,
        device = torch.device("cpu"),
    ):
        # Hardware related parameters.
        # These will be used directly internally.
        self.torch_dtype = torch_dtype
        self.device = device

        # Diffusers pipeline.
        self.pipe = None
        
        # Load model weights.
        self.pipe = diffusers.StableDiffusionPipeline.from_single_file(
            model_dir,
            torch_dtype = torch_dtype,
        )

        # Safety checker.
        self.pipe.safety_checker = safety_checker

        # Load LoRA weights.
        if lora_dir is not None:
            self.pipe.load_lora_weights(lora_dir)

        # CLIP skip.
        clip_layers = self.pipe.text_encoder.text_model.encoder.layers
        if clip_skip > 0:
            self.pipe.text_encoder.text_model.encoder.layers = clip_layers[:-clip_skip]

        # Scheduler.
        if scheduler in ["EulerAncestralDiscreteScheduler", "EADS"]:
            self.pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
        elif scheduler in ["EulerDiscreteScheduler", "EDS"]:
            self.pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
        elif scheduler in ["DPMSolverMultistepScheduler", "DPMSMS"]:
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

        # Device.
        self.pipe = self.pipe.to(self.device)

        # Prompt and negative prompts.
        self.prompt = None
        self.negative_prompt = None
        self.prompt_embeddings = None
        self.negative_prompt_embeddings = None
        self.set_prompts(
            prompt, 
            negative_prompt, 
            use_prompt_embeddings, 
            split_character
        )

    # %%
    def set_prompts(
        self, 
        prompt = None, 
        negative_prompt = None,
        use_prompt_embeddings = True,
        split_character = ",",
    ):
        """Set prompt and negative prompts.
        Optionally calculates the prompt embeddings.
        """
        if prompt is not None:
            self.prompt = prompt
        if negative_prompt is not None:
            self.negative_prompt = negative_prompt

        # Currently making prompt embeddings require both the prompt and
        # the negative prompt to be set.
        if type(self.prompt) == str and type(self.negative_prompt) == str:
            if use_prompt_embeddings is True:
                self.get_prompt_embeddings(split_character = split_character)

    # %%
    def get_prompt_embeddings(
        self, 
        split_character = ",", 
        return_embeddings = False,
    ):
        """Prompt embeddings to overcome CLIP 77 token limit.
        https://github.com/huggingface/diffusers/issues/2136
        """

        max_length = self.pipe.tokenizer.model_max_length
  
        # Simple method of checking if the prompt is longer than the negative
        # prompt - split the input strings using `split_character`.
        count_prompt = len(self.prompt.split(split_character))
        count_negative_prompt = len(self.negative_prompt.split(split_character))

        # If prompt is longer than negative prompt.
        if count_prompt >= count_negative_prompt:
            input_ids = self.pipe.tokenizer(
                self.prompt, return_tensors = "pt", truncation = False,
            ).input_ids.to(self.device)
            shape_max_length = input_ids.shape[-1]
            negative_ids = self.pipe.tokenizer(
                self.negative_prompt, 
                truncation = False, 
                padding = "max_length", 
                max_length = shape_max_length,
                return_tensors = "pt",
            ).input_ids.to(self.device)
        # If negative prompt is longer than prompt.
        else:
            negative_ids = self.pipe.tokenizer(
                self.negative_prompt, return_tensors = "pt", truncation = False,
            ).input_ids.to(self.device)
            shape_max_length = negative_ids.shape[-1]
            input_ids = self.pipe.tokenizer(
                self.prompt,
                return_tensors = "pt",
                truncation = False,
                padding = "max_length",
                max_length = shape_max_length,
            ).input_ids.to(self.device)

        # Concatenate the individual prompt embeddings.
        concat_embeds = []
        neg_embeds = []
        for i in range(0, shape_max_length, max_length):
            concat_embeds.append(
                self.pipe.text_encoder(input_ids[:, i: i + max_length])[0]
            )
            neg_embeds.append(
                self.pipe.text_encoder(negative_ids[:, i: i + max_length])[0]
            )
        
        self.prompt_embeddings = torch.cat(concat_embeds, dim = 1)
        self.negative_prompt_embeddings = torch.cat(neg_embeds, dim = 1)

        if return_embeddings is True:
            return torch.cat(concat_embeds, dim = 1), torch.cat(neg_embeds, dim = 1)

    # %%
    def run_pipe(
        self,
        steps = 50,
        width = 512,  
        height = 768, 
        scale = 7.0,
        seed = 0,
        use_prompt_embeddings = False,
        verbose = False,
    ):
        """Runs the loaded model.
        """
        if self.prompt is None and self.prompt_embeddings is None:
            return 
        
        if self.pipe is None:
            return
        
        start_time = time.time()

        if use_prompt_embeddings is True:
            imgs = self.pipe(
                prompt_embeds = self.prompt_embeddings,
                negative_prompt_embeds = self.negative_prompt_embeddings,
                width = width,
                height = height,
                guidance_scale = scale,
                num_inference_steps = steps,
                num_images_per_prompt = 1,
                generator = torch.manual_seed(seed),
            ).images
        else:
            imgs = self.pipe(
                prompt = self.prompt,
                negative_prompt = self.negative_prompt,
                width = width,
                height = height,
                guidance_scale = scale,
                num_inference_steps = steps,
                num_images_per_prompt = 1,
                generator = torch.manual_seed(seed),
            ).images

        end_time = time.time()
        time_elapsed = end_time - start_time

        if verbose is True:
            sys.stdout.write("{:.2f}s.\n".format(time_elapsed));

        return imgs[0]