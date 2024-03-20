import sys
import time
import compel
import diffusers
import torch


# %%
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
        use_compel = True,
        torch_dtype = torch.float32,
        device = torch.device("cpu")
    ):
        """
        Text2Image stable diffusion pipeline capable of handling:
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
                Set to None to remove turn safety checker off.
                Can also use customized safety checkers.
            use_prompt_embeddings: bool
                If True, prompt embeddings and negative prompt embeddings will be 
                used instead. Overcomes CLIP's 77 token limit.
            use_compel: True
                If True, use Compel to make prompt embeddings. True by default.
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

        # Load CivitAI model weights in the form of safetensors.
        # TODO add support for other file types or for downloading models from HuggingFace.
        self.load_model(model_dir, torch_dtype)

        # Safety checker.
        self.pipe.safety_checker = safety_checker

        # Load LoRA weights.
        # TODO clean this up!
        if len(lora_dirs) == 0:
            if lora_dir is not None:
                self.pipe.load_lora_weights(lora_dir)
        else:
            for ldir, lsc in zip(lora_dirs, lora_scales):
                self.pipe.load_lora_weights(ldir)
                self.pipe.fuse_lora(lora_scale = lsc)

        # CLIP skip.
        self.clip_skip(clip_skip)

        # Textual inversion.
        # https://huggingface.co/docs/diffusers/v0.24.0/en/using-diffusers/loading_adapters#textual-inversion
        self.set_textual_inversion(textual_inversion_dirs, textual_inversion_tokens)

        # Scheduler.
        # https://huggingface.co/docs/diffusers/v0.26.2/en/api/schedulers/overview#schedulers
        self.set_scheduler(scheduler, scheduler_configs)

        # Device.
        self.pipe = self.pipe.to(self.device)

        # Prompt and negative prompts.
        self.prompt = None
        self.negative_prompt = None
        self.prompt_embeddings = None
        self.negative_prompt_embeddings = None
        self.compel = compel.Compel(tokenizer = self.pipe.tokenizer, text_encoder = self.pipe.text_encoder)
        self.set_prompts(prompt, negative_prompt, use_prompt_embeddings, use_compel)

    # %% 
    def load_model(self, model_dir, torch_dtype):
        self.pipe = diffusers.StableDiffusionPipeline.from_single_file(model_dir, torch_dtype = torch_dtype)

    # %%
    def clip_skip(self, clip_skip = 0):
        if clip_skip > 0:
            clip_layers = self.pipe.text_encoder.text_model.encoder.layers
            self.pipe.text_encoder.text_model.encoder.layers = clip_layers[:-clip_skip]

    # %%
    def set_textual_inversion(self, textual_inversion_dirs = [], textual_inversion_tokens = []):
        for tid, tis in zip(textual_inversion_dirs, textual_inversion_tokens):
            self.pipe.load_textual_inversion(tid, token = tis)


    # %%
    def set_scheduler(self, scheduler = None, scheduler_configs = None):
        if scheduler_configs is None or len(scheduler_configs) == 0:
            scheduler_configs = self.pipe.scheduler.config
        else:
            for k in self.pipe.scheduler.config.keys():
                scheduler_configs[k] = self.pipe.scheduler.config.get(k)
        
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
        use_compel = True
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
                if use_compel is True:
                    self.get_compel_prompt_embeddings()
                else:
                    self.get_prompt_embeddings()

    # %% 
    def get_compel_prompt_embeddings(self, return_embeddings = False):
        """Use Compel to generate prompt embeddings to overcome CLIP 77 token limit.
        https://huggingface.co/docs/diffusers/en/using-diffusers/weighted_prompts
        """
        self.prompt_embeddings = self.compel([self.prompt])
        self.negative_prompt_embeddings = self.compel([self.negative_prompt])

        if return_embeddings is True:
            return self.prompt_embeddings, self.negative_prompt_embeddings
                    
    # %%
    def get_prompt_embeddings(self, return_embeddings = False):
        """Prompt embeddings to overcome CLIP 77 token limit.
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
        use_prompt_embeddings = False,
        verbose = False
    ):
        """Runs the loaded model for 1 image.
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
