import sys
import os
import time
import compel
import diffusers
import torch

# This repository wraps code from the diffusers library.
# https://huggingface.co/docs/diffusers/en/using-diffusers/write_own_pipeline
# The StableDiffusionPipeline consists of the followinng seven components:
# 1. "feature_extractor": a CLIPImageProcessor for encoding images.
# 2. "safety_checker": a component for screening against harmful content.
# 3. "scheduler": an instance of a scheduler.
# 4. "text_encoder": a CLIPTextModel for encoding text.
# 5. "tokenizer": a CLIPTokenizer for tokenizing text.
# 6. "unet": a denoising U-Net in latent space.
# 7. "vae": an encoder/decoder for encoding/decoding between latent and RGB space.

# %%
# TODO update code to use the latest versions of diffusers.
class Text2ImagePipe(object):
    # %%
    def __init__(
        self,
        model_dir,
        prompt = None,
        negative_prompt= None,
        scheduler = None,
        scheduler_configs = None,
        lora_dir = None,
        lora_dirs = [],
        lora_scales = [],
        clip_skip = 0,
        textual_inversion_dirs = [],
        textual_inversion_tokens = [],
        safety_checker = None,
        use_prompt_embeddings = True,
        use_compel = False,
        img2img = False,
        torch_dtype = torch.float32,
        device = torch.device("cpu")
    ):
        """Text2Image stable diffusion pipeline capable of handling:
        1. Prompt and negative prompt embeddings.
        2. Loading LoRAs.
        3. CLIP skips.
        4. Safety checker.

        Inputs:
            model_dir: str
                Path to the model checkpoint safetensors file.
            prompt: str
                Prompt.
            negative_prompt: str
                Negative prompt.
            scheduler: str
                Scheduler to use. Choose from: EADS, EDS or DPMSMS.
            scheduler_configs: dict
                Scheduler configurations in the form of a dict.
            lora_dir: str
                Path to a single LoRA safetensors file.
            lora_dirs: list of str
                Paths to multiple LoRA safetensors files.
            lora_scales: list of floats
                Corresponding scaling factors for the LoRAs in lora_dirs.
            clip_skip: int
                Number of CLIP layers to skip. 0 means no CLIP skipping.
            textual_inversion_dirs: list
                Paths to multiple textual inversion embedding files.
            textual_inversion_tokens: list
                List of tokens corresponding to the textual inversion embedding files.
            safety_checker: None
                Set to None to remove turn safety checker off. None by default
                Can also use customized safety checkers as argument.
            use_prompt_embeddings: bool
                If True, prompt embeddings and negative prompt embeddings will be 
                used instead. Overcomes CLIP's 77 token limit.
            use_compel: bool
                If True, Compel will be used to handle prompt embeddings. False by default.
            img2img: False
                If True, the img2img pipeline will be loaded instead of the default text2img pipeline.
            torch_dtype: torch.float32 or torch.float16.
                Use torch.float32 if using torch.device("cpu"), and
                use torch.float16 if using torch.device("cuda").
            device: torch.device("cpu") or torch.device("cuda")
                Use CUDA if you have access to a GPU! Makes life easier.
        """
        # Hardware related parameters.
        # These will be used directly internally.
        self.torch_dtype = torch_dtype
        self.device = device

        # Diffusers pipeline.
        self.pipe = None
        self.img2img = img2img # Set to True for image-to-image pipeline.

        # Load CivitAI model weights in the form of safetensors into the StableDiffusionPipeline.
        self.load_model_weights(model_dir, torch_dtype)

        # Safety checker. Turn off for more control.
        self.pipe.safety_checker = safety_checker

        # Load and fuse LoRA weights.
        # https://huggingface.co/docs/diffusers/v0.24.0/en/using-diffusers/loading_adapters#lora
        self.load_lora_weights(lora_dir=lora_dir, lora_dirs=lora_dirs, lora_scales=lora_scales)

        # CLIP skip.
        self.clip_skip(clip_skip)

        # Scheduler.
        # https://huggingface.co/docs/diffusers/v0.26.2/en/api/schedulers/overview#schedulers
        self.set_scheduler(scheduler, scheduler_configs)

        # Device.
        self.pipe = self.pipe.to(self.device)

        # Textual inversion.
        # https://huggingface.co/docs/diffusers/v0.24.0/en/using-diffusers/loading_adapters#textual-inversion
        self.load_textual_inversion_weights(textual_inversion_dirs, textual_inversion_tokens)

        # Create prompt and negative prompt embeddings.
        self.prompt = None
        self.negative_prompt = None
        self.prompt_embeddings = None
        self.negative_prompt_embeddings = None
        if use_compel is True:
            if len(textual_inversion_dirs) > 0:
                textual_inversion_manager = compel.DiffusersTextualInversionManager(self.pipe)
            else:
                textual_inversion_manager = None
            self.compel = compel.Compel(tokenizer=self.pipe.tokenizer, 
                                        text_encoder=self.pipe.text_encoder,
                                        textual_inversion_manager=textual_inversion_manager, 
                                        truncate_long_prompts=False)
        else:
            self.compel = None
        self.set_prompts(prompt, negative_prompt, use_prompt_embeddings, use_compel)

    # %% 
    def load_model_weights(self, model_dir, torch_dtype):
        """Loads a pre-trained model into the StableDiffusionPipeline. Able to load .safetensors files
        and diffusers model directories.
        https://huggingface.co/docs/diffusers/v0.24.0/en/using-diffusers/using_safetensors

        Inputs:
            model_dir: str
                Path to the pre-trained model safetensors file or a diffusers model directory.
            torch_dtype: torch.dtype
                GPU floating point size - torch.float32 or torch.float16.
        """
        if os.path.splitext(model_dir)[-1] == ".safetensors":
            # Load .safetensors file.
            if self.img2img is True:
                self.pipe = diffusers.StableDiffusionImg2ImgPipeline.from_single_file(model_dir, torch_dtype=torch_dtype)
            else:
                self.pipe = diffusers.StableDiffusionPipeline.from_single_file(model_dir, torch_dtype=torch_dtype)
        else:
            # Load diffusers model repo.
            if self.img2img is True:
                self.pipe = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(model_dir, torch_dtype=torch_dtype)
            else:
                self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch_dtype)
            
    # %%
    def load_lora_weights(self, lora_dir = None, lora_dirs = [], lora_scales = []):
        """Loads one or more pre-trained LoRA weights into the StableDiffusionPipeline.
        For multiple LoRA weights, a scaling factor is applied during the LoRA fusing process.
        """
        # TODO clean this up to handle single and multiple LoRA dirs with the same input argument!
        if len(lora_dirs) == 0:
            # Load a single LoRA.
            if lora_dir is not None:
                self.pipe.load_lora_weights(lora_dir)
        else:
            # Load and fuse multiple LoRAs.
            for ldir, lsc in zip(lora_dirs, lora_scales):
                self.pipe.load_lora_weights(ldir)
                self.pipe.fuse_lora(lora_scale = lsc)

    # %%
    def clip_skip(self, clip_skip = 0):
        """Number of CLIP layers to skip. This parameter value is slightly differently from the commonly
        used one: 0 means no CLIP skipping will be used, 1 means the final 1 CLIP layers will be skipped,
        2 means the final 2 CLIP layers will be skipped and so on.

        Inputs:
            clip_skip: int
                Number of CLIP layers to skip. 0 means no CLIP skipping will be used.
        """
        if clip_skip > 0:
            clip_layers = self.pipe.text_encoder.text_model.encoder.layers
            self.pipe.text_encoder.text_model.encoder.layers = clip_layers[:-clip_skip]

    # %%
    def load_textual_inversion_weights(self, textual_inversion_dirs = [], textual_inversion_tokens = []):
        """Loads textual inversion pre-trained model weights into the StableDiffusionPipeline.
        Textual inversion is a technique for learning a specific concept from some images which you can use to 
        generate new images conditioned on that concept.
        https://huggingface.co/docs/diffusers/v0.24.0/en/using-diffusers/loading_adapters#textual-inversion
        https://huggingface.co/docs/diffusers/en/using-diffusers/weighted_prompts#textual-inversion
        
        Inputs:
            textual_inversion_dirs: list
                List of paths to the pre-trained textual inversion models.
            textual_inversion_tokens:
                Optional list of corresponding textual inversion tokens. Specify tokens for more control.
        """
        if len(textual_inversion_dirs) > len(textual_inversion_tokens):
            diff = [None] * (len(textual_inversion_dirs) - len(textual_inversion_tokens))
            textual_inversion_tokens = textual_inversion_tokens + diff

        for tid, tis in zip(textual_inversion_dirs, textual_inversion_tokens):
            if tis is None:
                # Positive textual inversion. No special token. Less control over the output.
                self.pipe.load_textual_inversion(tid)
            else:
                # Negative textual inversion. Requires a special token. More control over the output.
                self.pipe.load_textual_inversion(tid, token = tis)

    # %%
    def set_scheduler(self, scheduler = None, scheduler_configs = None):
        """Loads a scheduler into the pipeline.
        https://huggingface.co/docs/diffusers/v0.24.0/en/using-diffusers/loading#schedulers
        
        Inputs:
            scheduler: str
                Scheduler name.
            scheduler_configs: dict
                Dictionary of scheduler configurations.
        """
        if scheduler_configs is None or len(scheduler_configs) == 0:
            scheduler_configs = self.pipe.scheduler.config
        else:
            for k in self.pipe.scheduler.config.keys():
                scheduler_configs[k] = self.pipe.scheduler.config.get(k)
        # TODO add more schedulers.
        if scheduler in ["EulerAncestralDiscreteScheduler", "EADS"]:
            self.pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(scheduler_configs)
        elif scheduler in ["EulerDiscreteScheduler", "EDS"]:
            self.pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_configs)
        elif scheduler in ["DPMSolverMultistepScheduler", "DPMSMS"]:
            self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(scheduler_configs)

    # %%
    def set_prompts(
        self,
        prompt = None,
        negative_prompt = None,
        use_prompt_embeddings = True,
        use_compel = False
    ):
        """Set prompt and negative prompts. Optionally calculates the prompt embeddings.
        Also optionally uses the compel library to calculate the prompt embeddings.
        https://huggingface.co/docs/diffusers/en/using-diffusers/weighted_prompts#weighting
        
        Inputs:
            prompt: str
            negative_prompt: str
            use_prompt_embeddings: bool
            use_compel: bool
        """
        if prompt is not None:
            self.prompt = prompt
        if negative_prompt is not None:
            self.negative_prompt = negative_prompt

        # Currently making prompt embeddings require both the prompt and
        # the negative prompt to be set.
        if type(self.prompt) == str and type(self.negative_prompt) == str:
            if use_prompt_embeddings is True:
                if use_compel is True:
                    self.get_compel_prompt_embeddings()
                else:
                    self.get_prompt_embeddings()

    # %% 
    def get_compel_prompt_embeddings(self, return_embeddings = False):
        """Use Compel to generate prompt embeddings to overcome CLIP 77 token limit.
        https://huggingface.co/docs/diffusers/en/using-diffusers/weighted_prompts#weighting
        """
        prompt_embeddings = self.compel([self.prompt])
        negative_prompt_embeddings = self.compel([self.negative_prompt])
        # The pipeline requires that both embeddings be of the same length.
        [self.prompt_embeddings, self.negative_prompt_embeddings] = self.compel.pad_conditioning_tensors_to_same_length(
            [prompt_embeddings, negative_prompt_embeddings]
        )
        if return_embeddings is True:
            return self.prompt_embeddings, self.negative_prompt_embeddings
                    
    # %%
    def get_prompt_embeddings(self, return_embeddings = False):
        """Prompt embeddings to overcome CLIP 77 token limit. Does not use compel.
        https://github.com/huggingface/diffusers/issues/2136
        """
        max_length = self.pipe.tokenizer.model_max_length

        input_ids = self.pipe.tokenizer(
            self.prompt, return_tensors = "pt", truncation = False
        ).input_ids.to(self.device)
        negative_ids = self.pipe.tokenizer(
            self.negative_prompt, return_tensors = "pt", truncation = False
        ).input_ids.to(self.device)

        if input_ids.shape[-1] >= negative_ids.shape[-1]:
            shape_max_length = input_ids.shape[-1]
            negative_ids = self.pipe.tokenizer(
                self.negative_prompt, return_tensors = "pt", truncation = False, 
                padding = "max_length", max_length = shape_max_length
            ).input_ids.to(self.device)
        else:
            shape_max_length = negative_ids.shape[-1]
            input_ids = self.pipe.tokenizer(
                self.prompt, return_tensors = "pt", truncation = False, 
                padding = "max_length", max_length = shape_max_length
            ).input_ids.to(self.device)

        concat_embeds = []
        neg_embeds = []
        for i in range(0, shape_max_length, max_length):
            concat_embeds.append(self.pipe.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(self.pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

        self.prompt_embeddings = torch.cat(concat_embeds, dim = 1)
        self.negative_prompt_embeddings = torch.cat(neg_embeds, dim = 1)

        if return_embeddings is True:
            return self.prompt_embeddings, self.negative_prompt_embeddings

    # %%
    def run_pipe(
        self,
        steps = 50,
        width = 512,
        height = 768,
        scale = 7.0,
        seed = 0,
        image = None,
        strength = 0.8,
        use_prompt_embeddings = True,
        verbose = False
    ):
        """Runs the loaded model for 1 image.

        Inputs:
            steps: int
                Number of inference steps.
            width: int 
                Output image width.
            height: int
                Output image height.
            scale: float
                Guidance scale.
            seed: int
                RNG seed. Used to control outputs.
            image: Image
                Input image for img2img pipelines.
            strength: float
                Strength for img2img pipelines.
            use_prompt_embeddings: bool
                Use prompt embeddings or not. True by default as most prompts will be > 77 tokens long.
            verbose: bool
                Verbosity mode. Set to True for verbose mode.
        Returns:
            imgs: Image
                Generated image.
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
                image = image,
                strength = strength,
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
                image = image,
                strength = strength,
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
