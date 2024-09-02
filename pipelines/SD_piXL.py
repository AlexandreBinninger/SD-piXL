from typing import Union, AnyStr, List
import itertools
import math
from datetime import datetime
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import diffusers
from tqdm.auto import tqdm
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf

import torch
import torch.nn.functional as nnf
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

from transformers import AutoProcessor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from diffusers.optimization import get_scheduler

from libs.engine import ModelState
from libs.engine import write_config_to_yaml

from pipelines import LSDSSDXLControlNetPipeline
from pipelines.utils import view_images, euclidean_similarity_target, euclidean_similarity_target_cls, cosine_similarity_create_target_cls
from pipelines.distillation import init_diffusion_pipeline, model2res
from models.pixelart import PixelArt

from utils import kmeans_downsample, palette_downsample, nearest_downsample, total_variation_loss, bilateral_edge_preserving_loss, Laplacian_Loss, FFT_Loss, gradient_loss
from utils import DepthEstimator, canny_edge
from utils.files_utils import load_hex


### SD_piXL ###

class SD_piXL(ModelState):
    def __init__(self, args):
        """
            Initialize the SD_piXL, 
        """
        args.batch_size = 1
        logdir_ = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}" \
                  f"-{args.diffusion.model_id}" \
                  f"-im{args.generator.image_H}x{args.generator.image_W}" \
                  f"-{args.generator.palette.split('/')[-1].split('.')[0]}"
        super().__init__(args, log_path_suffix=logdir_)
        
        # Save config
        write_config_to_yaml(self.args.yaml_config, self.results_path / "config.yaml")
        
        # Create directories to log intermediate results
        self.png_logs_dir = self.results_path / "png_logs"
        self.entropy_dir = self.results_path / "entropy"
        self.sd_sample_dir = self.results_path / 'sd_samples'
        self.tensorboard_dir = self.results_path / "tensorboard"
        self.laplacian_dir = self.results_path / "laplacian"
        self.grad_dir = self.results_path / "gradients"
        if self.accelerator.is_main_process:
            self.png_logs_dir.mkdir(parents=True, exist_ok=True)
            if (self.args.generator.initialize_renderer) and (self.args.image is None):            
                self.sd_sample_dir.mkdir(parents=True, exist_ok=True)
            if self.args.training.laplacian_scale > 0 and self.args.verbose:
                self.laplacian_dir.mkdir(parents=True, exist_ok=True)
            if self.args.verbose:
                self.grad_dir.mkdir(parents=True, exist_ok=True)
                self.entropy_dir.mkdir(parents=True, exist_ok=True)
                self.summary_writer = SummaryWriter(self.tensorboard_dir.as_posix())
        self.select_fpth = self.results_path / 'select_sample.png'
        
        
        # Load models
        if args.diffusion.model_id == "sdxl" or args.diffusion.model_id == "ssd1b":
            self.custom_pipeline = LSDSSDXLControlNetPipeline
        else:
            return NotImplementedError
        self.custom_scheduler = diffusers.DDPMScheduler
        self.cross_attention_kwargs = {
            "scale": args.diffusion.lora_scale
        }
        self.diffusion = init_diffusion_pipeline(
            args.diffusion.model_id,
            vae_id=args.diffusion.vae_id,
            custom_pipeline=self.custom_pipeline,
            custom_scheduler=self.custom_scheduler,
            controlnets_id = self.args.controlnet.models_id,
            device=self.device,
            local_files_only=not args.download,
            force_download=args.force_download,
            ldm_speed_up=args.diffusion.ldm_speed_up,
            enable_xformers=args.diffusion.enable_xformers,
            gradient_checkpoint=args.diffusion.gradient_checkpoint,
            lora_path=args.diffusion.lora_path,
            grayscale_prob=args.training.augmentation.grayscale_prob,
            hflip_prob=args.training.augmentation.hflip_prob,
            distorsion_prob=args.training.augmentation.distorsion_prob,
            distorsion_scale=args.training.augmentation.distorsion_scale,
        )
        self.g_device = torch.Generator(device=self.device).manual_seed(args.seed)
        self.depth_estimator = DepthEstimator(self.device)
        self.laplacian_loss = Laplacian_Loss(self.device, self.args.training.laplacian_kernel, self.args.training.laplacian_sigma, self.args.training.laplacian_mode)
        self.FFTLoss = FFT_Loss(self.device, self.args.generator.image_H, self.args.generator.image_W)

    def target_file_preprocess(self, tar_pil: Image.Image, image_size : int):
        process_comp = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ])
        target_img = process_comp(tar_pil)  # preprocess
        target_img = target_img.to(self.device)
        return target_img

    def get_t_range(self):
        if self.args.sd.sampling_method_t == "uniform":
            t_min, t_max = self.args.sd.t_min, self.args.sd.t_max
        elif self.args.sd.sampling_method_t == "linear":
            t_min = self.args.sd.t_max - (self.args.sd.t_max - self.args.sd.t_min) * (self.step+1) / self.args.training.steps
            t_max = self.args.sd.t_max - (self.args.sd.t_max - self.args.sd.t_min) * (self.step) / self.args.training.steps
        elif self.args.sd.sampling_method_t == "bounded_max":
            t_min = self.args.sd.t_min
            max_steps_t_bounds = self.args.sd.t_bound_reached * self.args.training.steps
            if self.step >= max_steps_t_bounds:
                t_max = self.args.sd.t_bound_max
            else:
                t_max = self.args.sd.t_max - (self.args.sd.t_max - self.args.sd.t_bound_max) * (self.step) / max_steps_t_bounds
        else:
            raise NotImplementedError
        return t_min, t_max
    
    def get_tau(self):
        if self.args.training.augmentation.random_tau:
            tau_min = self.args.training.augmentation.random_tau_min
            tau_max = self.args.training.augmentation.random_tau_max
            return np.random.uniform(tau_min, tau_max)
        else:
            return self.args.generator.tau

    @torch.no_grad()
    def caption_image(self, image_path):
        blip_processor = AutoProcessor.from_pretrained(self.args.caption.blip_model_id)
        blip = Blip2ForConditionalGeneration.from_pretrained(self.args.caption.blip_model_id)
        image = Image.open(image_path).convert("RGB")
        inputs = blip_processor(image, self.args.caption.query, return_tensors="pt").to(blip.device, torch.float16)
        generated_id = blip.generate(**inputs, min_new_tokens=self.args.caption.min_new_tokens, max_new_tokens=self.args.caption.max_new_tokens)[0]
        generated_text = blip_processor.decode(generated_id, skip_special_tokens=self.args.caption.skip_special_tokens).strip()
        del blip_processor
        del blip
        return generated_text
    
    @torch.no_grad()
    def rejection_sampling(self, img_caption: Union[AnyStr, List], diffusion_samples: List):
        import clip
        
        clip_model, preprocess = clip.load("ViT-B/32", device=self.device)

        # Encode text and images
        text_input = clip.tokenize([img_caption]).to(self.device)
        text_features = clip_model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        clip_images = torch.stack([
            preprocess(sample) for sample in diffusion_samples]
        ).to(self.device)
        image_features = clip_model.encode_image(clip_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # clip score to get the image that is most similar to the text
        similarity_scores = (text_features @ image_features.T).squeeze(0)

        selected_image_index = similarity_scores.argmax().item()
        selected_image = diffusion_samples[selected_image_index]
        return selected_image
    
    def diffusion_sampling(self):
        """Sampling images from SDXL"""
        from diffusers import DiffusionPipeline, AutoencoderTiny
        diffusion_samples = []
        self.diffusion.to("cpu")
        
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        height = width = model2res(self.args.diffusion.model_id)
        for i in range(self.args.diffusion.num_references):
            output = pipe(
                prompt=[self.args.prompt],
                negative_prompt=self.args.negative_prompt,
                height=height,
                width=width,
                num_images_per_prompt=1,
                num_inference_steps=self.args.diffusion.num_inference_steps,
                guidance_scale=self.args.diffusion.guidance_scale,
                generator=self.g_device
            )
            outputs_np = [np.array(img) for img in output.images]
            view_images(outputs_np, save_image=True, fp=self.sd_sample_dir / f'samples_{i}.png')
            diffusion_samples.extend(output.images)

        self.print(f"num_generated_samples: {len(diffusion_samples)}, shape: {outputs_np[0].shape}")

        del pipe
        torch.cuda.empty_cache()
        self.diffusion.to(self.device)
        return diffusion_samples
    
    def diffusion_target(self):
        """
        Sample images from diffusion model
        Then select one image from the sampled images using CLIP score
        Inspired by VectorFusion
        """
        # Sampling K images
        diffusion_samples = self.diffusion_sampling()

        # rejection sampling
        select_target = self.rejection_sampling(self.args.prompt, diffusion_samples)
        select_target_pil = Image.fromarray(np.asarray(select_target))  # numpy to PIL
        select_target_pil.save(self.select_fpth)
        torch.cuda.empty_cache()

        # load target file
        im_size = self.args.sd.im_size if self.args.sd.im_size else model2res(self.args.diffusion.model_id)
        assert self.select_fpth.exists(), f"{self.select_fpth} does not exist!"
        sampled_pil = Image.open(self.select_fpth.as_posix()).convert("RGB")
        sampled_img = self.target_file_preprocess(sampled_pil, im_size)
        self.print(f"load tfarget file from: {self.select_fpth.as_posix()}")
        return sampled_img

    def resize(self, img, size):
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = nnf.interpolate(img, size = size, mode=self.args.training.resize_mode)
        return img

    def save_render(self, renderer, soft : bool, gumbel = False, tau = 1.0, name : str = ""):
        img = renderer(smooth_softmax = soft, reg = self.args.generator.softmax_regularizer, gumbel = gumbel, tau = tau)
        img = img.detach().cpu().numpy() * 255
        img = np.round(img).astype('uint8')
        img = Image.fromarray(img)
        if 256 % img.size[0] == 0 and 256 % img.size[1] == 0:
            resize_size = (max(self.args.saving_resize, img.size[0]), max(self.args.saving_resize, img.size[1]))
            img = img.resize(resize_size, Image.NEAREST)
        img.save(self.png_logs_dir / f"{name}.png")

    def get_condition_controlnet(self, controlnet_type, image, im_size):
        if controlnet_type in ["canny_small", "canny_mid", "canny"]:
            return canny_edge(image, self.args.controlnet.canny_threshold1, self.args.controlnet.canny_threshold2, self.args.controlnet.canny_blur, self.args.controlnet.canny_blur_radius)
        elif controlnet_type in ["depth_small", "depth_mid", "depth"]:
            return self.depth_estimator.get_depth_map(image, size=(im_size, im_size), mode="bicubic")
        

    def pixelization(self, target_img):
        """
            Pixelize the target image using a classic method. Used notably for initialization of the renderer.
        """
        if self.args.generator.initialization_method == "nearest":
            target_pixelated, target_cls_index, palette = nearest_downsample(target_img, self.args.generator.image_H, self.args.generator.image_W)
        elif self.args.generator.initialization_method == "kmeans":
            target_pixelated, target_cls_index, palette = kmeans_downsample(target_img, self.args.generator.image_H, self.args.generator.image_W, self.args.generator.kmeans_nb_colors)
        elif self.args.generator.initialization_method == "palette":
            palette = load_hex(self.args.generator.palette)
            target_pixelated, target_cls_index, palette = palette_downsample(target_img, self.args.generator.image_H, self.args.generator.image_W, palette)
        elif self.args.generator.initialization_method == "palette-nearest":
            palette = load_hex(self.args.generator.palette)
            palette = torch.from_numpy(palette).to(target_img.device).byte()
            target_pixelated = nnf.interpolate(target_img, size=(self.args.generator.image_H, self.args.generator.image_W), mode='nearest')
        elif self.args.generator.initialization_method == "palette-bilinear":
            palette = load_hex(self.args.generator.palette)
            palette = torch.from_numpy(palette).to(target_img.device).byte()
            target_pixelated = nnf.interpolate(target_img, size=(self.args.generator.image_H, self.args.generator.image_W), mode='bilinear')
        else:
            raise NotImplementedError

        target_pixelated = target_pixelated.squeeze(0).permute(1, 2, 0)
        
        if self.args.generator.initialization_method not in ["palette-bilinear", "palette-nearest"]:
            if self.args.generator.init_distance == "cosine_similarity":
                target_cls = cosine_similarity_create_target_cls(target_cls_index, palette) * math.sqrt(palette.shape[0])
            elif self.args.generator.init_distance in ["l2", "l1"]:
                target_cls = euclidean_similarity_target_cls(target_cls_index, palette, distance_type=self.args.generator.init_distance) * math.sqrt(palette.shape[0])
            elif self.args.generator.init_distance == "random":
                h, w = target_cls_index.shape
                n = palette.shape[0]
                target_cls = torch.rand((h, w, n), device=palette.device)
            elif self.args.generator.init_distance == "zero":
                h, w = target_cls_index.shape
                n = palette.shape[0]
                target_cls = torch.zeros((h, w, n), device=palette.device)
            else:
                raise NotImplementedError
        else:
            target_cls, target_pixelated = euclidean_similarity_target(target_pixelated, palette, distance_type=self.args.generator.init_distance)
            target_cls *= math.sqrt(palette.shape[0])

        init_pixelized_image = Image.fromarray((target_pixelated.detach().cpu().numpy()*255).astype('uint8'))
        if self.args.verbose:
            init_pixelization_path = self.results_path / "init_pixelization.png"
            init_pixelized_image.save(init_pixelization_path)
        return target_cls, palette
    
    def run(self):
        """
            Run SD-piXL 
        """
        
        im_size = self.args.sd.im_size if self.args.sd.im_size else model2res(self.args.diffusion.model_id)
        
        ## Initialization
        with torch.no_grad():
            if self.args.automatic_caption:
                if self.args.image is None:
                    raise ValueError("/!\ Contradicting arguments.\nAn image is required for automatic captioning. Check that the 'image' field is not empty, or provide a prompt and set the field automatic_caption to false.") 
                if self.args.verbose:
                    self.print("captioning image...")
                caption = self.caption_image(self.args.image)
                self.print(f"Automatic caption: {caption}")
                self.args.prompt = caption
                
            if self.args.generator.initialize_renderer or self.args.generator.initialization_method == "kmeans":
                if self.args.image is not None:
                    ## Using a reference image as input      
                    if self.args.verbose:
                        self.print("From scratch with Score Distillation Sampling...")
                        self.print(f"image: {self.args.image}\n")
                    target_pil = Image.open(self.args.image).convert("RGB")
                    target_img = self.target_file_preprocess(target_pil, im_size)
                else:
                    # Perform Text-to-image, then use the best generated image as input
                    self.print("Generating input image based on input prompt")
                    target_img = self.diffusion_target()
                    self.print("Done")
                
                ## Pixelize the target image for initialization
                target_cls, palette = self.pixelization(target_img)
            else:
                if self.args.verbose:
                    self.print("From scratch with Score Distillation Sampling...")
                target_pil = Image.fromarray(np.zeros((im_size, im_size, 3), dtype=np.uint8))
                target_img = self.target_file_preprocess(target_pil, im_size)
                target_cls = None
                palette = self.args.generator.palette
                
        if self.args.verbose:
            if target_img is not None:
                target_image = Image.fromarray((target_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()*255).astype('uint8'))
                target_image.save(self.results_path / f"target_image.png")
                print("Saving target image to ", self.results_path / f"target_image.png")
        
        ## Controlnet settings
        controlnet_imgs = []
        target_image = Image.fromarray((target_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()*255).astype('uint8'))
        for controlnet_type in self.args.controlnet.models_id:
            controlnet_img = self.get_condition_controlnet(controlnet_type, target_image, im_size)
            controlnet_imgs.append(controlnet_img)
            if self.args.verbose:
                controlnet_img.save(self.results_path / f"control_{controlnet_type}.png")

        ## Initialize renderer, optimizer and scheduler
        torch.cuda.empty_cache()
        renderer = self.load_renderer(target_cls, palette, self.args.generator.initialize_renderer)
        parameters = renderer.parameters()

        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.args.training.learning_rate,
            betas=(self.args.training.adam_beta1, self.args.training.adam_beta2),
            weight_decay=self.args.training.adam_weight_decay,
            eps=self.args.training.adam_epsilon,
        )
        lr_scheduler = get_scheduler(
            self.args.training.lr_scheduler,
            step_rules=self.args.training.lr_step_rules,
            optimizer=optimizer,
            num_warmup_steps=self.args.training.lr_warmup_steps * self.args.gradient_accumulate_step,
            num_training_steps=self.args.training.steps * self.args.gradient_accumulate_step,
            num_cycles=self.args.training.lr_cycles,
            power=2.0
        )

        self.step = 0
        self.print(f"\ntotal sds optimization steps: {self.args.training.steps}")

        renderer, optimizer, lr_scheduler, self.diffusion, self.diffusion.vae = self.accelerator.prepare(
            renderer, optimizer, lr_scheduler, self.diffusion, self.diffusion.vae
        )
        if self.args.verbose:
            self.save_render(renderer, soft=False, name = f"00_initialization")

        with self.accelerator.autocast():
            ## Initialize the pipeline for Score Distillation Sampling
            self.diffusion.preprocess_SDS(
                im_size=im_size,
                prompt=[self.args.prompt],
                negative_prompt=self.args.negative_prompt,
                cross_attention_kwargs = self.cross_attention_kwargs
            )
            
            ## Optimization loop
            with tqdm(initial=self.step, total=self.args.training.steps, disable=not self.accelerator.is_main_process) as pbar:
                while self.step < self.args.training.steps:
                    tau = self.get_tau()
                    rendered_raw = renderer(smooth_softmax = self.args.generator.smooth_softmax, reg = self.args.generator.softmax_regularizer, gumbel = self.args.generator.gumbel, tau = tau).to(self.device)
                    raster_img = self.resize(rendered_raw, (im_size, im_size))
                    t_min, t_max = self.get_t_range()
                    
                    ## Get the loss from the Score Distillation Sampling, used to get the gradients. Warning: The "grad" is only returned here for vizualization purposes. L_sds is used to optimize the renderer.
                    L_sds, grad, augmented_img, timestep = self.diffusion.score_distillation_sampling_preprocessed(
                        raster_img,
                        controlnet_imgs,
                        im_size=im_size,
                        guidance_scale=self.args.sd.guidance_scale,
                        grad_scale=self.args.sd.grad_scale,
                        t_range=[t_min, t_max],
                        use_controlnet=self.args.controlnet.use_controlnet,
                        control_guidance_start = self.args.controlnet.control_guidance_start,
                        control_guidance_end = self.args.controlnet.control_guidance_end,
                        controlnet_conditioning_scale = OmegaConf.to_object(self.args.controlnet.controlnet_conditioning_scale),
                        cross_attention_kwargs = self.cross_attention_kwargs,
                        w_mode = self.args.sd.w_mode
                    )
                    
                    ## Additional regularizations
                    L_tv = torch.tensor([0.0], device=self.device)
                    L_fft = torch.tensor([0.0], device=self.device)
                    L_bil = torch.tensor([0.0], device=self.device)
                    L_lap = torch.tensor([0.0], device=self.device)
                    L_grad = torch.tensor([0.0], device=self.device)
                    laplacian_img = None
                    if self.args.training.tv_scale > 0:
                        L_tv = total_variation_loss(rendered_raw) * self.args.training.tv_scale
                    if self.args.training.fft_scale > 0:
                        L_fft = self.FFTLoss(rendered_raw) * self.args.training.fft_scale
                    if self.args.training.bilateral_scale > 0:
                        L_bil = bilateral_edge_preserving_loss(rendered_raw, 
                                                               self.args.training.bilateral_sigma_color, 
                                                               self.args.training.bilateral_sigma_space, 
                                                               self.args.training.bilateral_max_distance) * self.args.training.bilateral_scale
                    if  self.args.training.laplacian_scale > 0:
                        L_lap, laplacian_img = self.laplacian_loss(rendered_raw)
                        L_lap = L_lap * self.args.training.laplacian_scale
                    if self.args.training.gradient_loss_scale > 0:
                        L_grad, grad_img_x, grad_img_y, grad_target_x, grad_target_y = gradient_loss(rendered_raw, target_img)
                        L_grad = L_grad * self.args.training.gradient_loss_scale
                    
                    ## Optimization
                    loss= (L_sds + L_tv + L_fft + L_bil + L_lap + L_grad)/self.args.gradient_accumulate_step
                    optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    if self.args.training.clip_grad:
                        params_to_clip = itertools.chain(renderer.parameters())
                        self.accelerator.clip_grad_norm_(params_to_clip, self.args.training.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()

                    # Centering of the weights
                    if self.accelerator.is_main_process:
                        with torch.no_grad():
                            renderer.balance()            
                            
                            # Saving intermediate information. Set args.verbose to true to save more.
                            if self.step % self.args.training.save_steps == 0:
                                self.print(f"Saving intermediate results, iteration {self.step}")
                                self.save_render(renderer, soft=False, gumbel = False, tau = tau, name = f"{self.step}_hard")
                                self.save_render(renderer, soft=True, gumbel = False, tau = tau, name = f"{self.step}_soft")
                                
                            # Periodic checkpoint save
                            if self.step > 0 and self.step % self.args.save_step == 0:
                                print(f"Saving checkpoint, iteration {self.step}")
                                save_path = self.results_path / f"checkpoint_{self.step}"
                                os.makedirs(save_path, exist_ok=True)
                                state_dict = {
                                    "renderer": renderer.state_dict(),
                                    "palette": self.args.generator.palette, # a str
                                    "height": self.args.generator.image_H,
                                    "width": self.args.generator.image_W,
                                }
                                torch.save(state_dict, save_path / "model_checkpoint.pth")
                                
                            if self.args.verbose:
                                # Saving intermediate results
                                if self.step % self.args.training.save_steps == 0:
                                    if self.args.generator.gumbel:
                                        self.save_render(renderer, soft=False, gumbel = True, tau = tau, name = f"{self.step}_hard_gumbel")
                                        self.save_render(renderer, soft=True, gumbel = True, tau = 1.0, name = f"{self.step}_soft_gumbel")
                                    img_augmented = Image.fromarray((augmented_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()*255).astype('uint8'))
                                    img_augmented.save(self.png_logs_dir / f"{self.step}_augmented.png")

                                # Get variance, entropy, max probability. Log to tensorboard
                                variance = renderer.get_probability_variance(take_mean = True)
                                variance = variance.detach().cpu().numpy()
                                variance = variance.mean()
                                self.summary_writer.add_scalar("average_variance", variance, self.step)

                                nb_colours = renderer.nb_col
                                normalized_variance = variance * (nb_colours**2)/(nb_colours-1)
                                self.summary_writer.add_scalar("average_normalized_variance", normalized_variance, self.step)

                                entropy = renderer.get_probability_entropy(take_mean = True)
                                entropy = entropy.detach().cpu().numpy()
                                entropy = entropy.mean()
                                self.summary_writer.add_scalar("average_entropy", entropy, self.step)
                                normalized_entropy = entropy / np.log(nb_colours)
                                self.summary_writer.add_scalar("average_normalized_entropy", normalized_entropy, self.step)

                                max_prob = renderer.get_probability_max(take_mean = True)
                                max_prob = max_prob.detach().cpu().numpy()
                                max_prob = max_prob.mean()
                                self.summary_writer.add_scalar("max_prob", max_prob, self.step)

                                ## Save entropy image
                                if self.step % self.args.training.save_steps == 0:
                                        # Get and process entropy image
                                        normalized_entropy_img = renderer.get_probability_entropy(take_mean=False) / np.log(nb_colours)
                                        normalized_entropy_img = normalized_entropy_img.detach().cpu().numpy()

                                        # Apply colormap and save
                                        plt.imsave(self.entropy_dir / f"{self.step}_normalized_entropy.png", normalized_entropy_img, cmap='coolwarm', format='png')
                                        
                                        # Save gradients
                                        grad_img = self.diffusion.vae.decode(grad / self.diffusion.vae.config.scaling_factor, return_dict=False)[0]
                                        grad_img = self.diffusion.image_processor.postprocess(grad_img, 'pil')[0]
                                        grad_img.save(self.grad_dir / f"{self.step}_grad_{timestep}.png")
                                        
                                        if laplacian_img is not None:
                                            laplacian_img = rearrange(laplacian_img, 'c h w -> h w c')
                                            laplacian_img = laplacian_img.abs().detach().cpu().numpy()
                                            plt.imsave(self.laplacian_dir / f"{self.step}_laplacian.png", laplacian_img, cmap='coolwarm', format='png')

                    ## Log info to the tqdm bar
                    pbar_str = f"lr: {lr_scheduler.get_last_lr()[0]}, L_total: {loss.item():.5e}"
                    if self.args.verbose:                        
                        pbar.set_description(
                            pbar_str +
                            f", L_tv: {L_tv.item():.5e}, L_fft: {L_fft.item():.5e}, L_bil: {L_bil.item():.5e}, L_lap: {L_lap.item():.5e}, L_grad: {L_grad.item():.5e} "+
                            f"sds: {grad.mean().item():.5e}, "+
                            f"variance: {variance:.5e}"
                        )
                    else:
                        pbar.set_description(pbar_str)
                    self.step += 1
                    pbar.update(1)
        
        self.save_render(renderer, soft=False, gumbel = False, tau = tau, name = "../final_argmax")
        self.save_render(renderer, soft=True, gumbel = False, tau = tau, name = "../final_softmax")

        self.close(msg="Pixel generation is complete.")

    def load_renderer(self,
                      target_cls = None,
                      palette = None,
                      initialize_renderer = True):
        if palette is None:
            palette = self.args.generator.palette
        renderer = PixelArt(self.args.generator.image_H, self.args.generator.image_W, palette, self.args.generator.std)
        if initialize_renderer and target_cls is not None:
            renderer.initialize(target_cls)
            if self.args.generator.init_distance != "zero":
                renderer.normalize_softmax(renderer.get_std(take_mean = True))
        return renderer
