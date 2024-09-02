from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torchvision import transforms
import diffusers
from diffusers import AutoencoderKL, AutoencoderTiny, ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from torchvision import transforms
import torchvision.transforms.functional as TF


class LSDSSDXLControlNetPipeline(StableDiffusionXLControlNetPipeline):
    def setup_augmentation(self, grayscale_prob, hflip_prob, distorsion_prob, distorsion_scale):
        self.grayscale_prob = grayscale_prob
        self.hflip_prob = hflip_prob
        self.distorsion_prob = distorsion_prob
        self.distorsion_scale = distorsion_scale

    def encode_(self, images):
        images = (2 * images - 1).clamp(-1.0, 1.0)  # images: [B, 3, H, W]
        if type(self.vae) == AutoencoderKL:
            latents = self.vae.encode(images).latent_dist.sample()
            latents = self.vae.config.scaling_factor * latents
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        elif type(self.vae) == AutoencoderTiny:
            latents = self.vae.encode(images).latents
            latents = self.vae.config.scaling_factor * latents
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def x_augment_same(self, images, img_size: int = 1024):
        # Apply the same transformations to all images
        # Generate random parameters for the transformations
        if torch.rand(1) < self.grayscale_prob:
            for i, image in enumerate(images):
                images[i] = TF.rgb_to_grayscale(image, num_output_channels=3)
        if torch.rand(1) < self.distorsion_prob:
            startpoints, endpoints = transforms.RandomPerspective.get_params(
                img_size, img_size, distortion_scale=self.distorsion_scale)
            for i, image in enumerate(images):
                images[i] = TF.perspective(
                    image, startpoints, endpoints, interpolation=TF.InterpolationMode.BILINEAR)

        if torch.rand(1) < self.hflip_prob:
            for i, image in enumerate(images):
                images[i] = TF.hflip(image)

        # Crop
        padding_mode = 'reflect'
        fill = 0
        size = [img_size, img_size]
        for k, image in enumerate(images):
            _, height, width = TF.get_dimensions(image)
            # pad the width if needed
            if width < size[1]:
                padding = [size[1] - width, 0]
                image = TF.pad(image, padding, fill, padding_mode)
            # pad the height if needed
            if height < size[0]:
                padding = [0, size[0] - height]
                image = TF.pad(image, padding, fill, padding_mode)
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(img_size, img_size))
            images[k] = TF.crop(image, i, j, h, w)
        return images

    def preprocess_SDS(self,
                       im_size: int,
                       prompt: Union[List, str],
                       prompt_2: Optional[Union[List, str]] = None,
                       negative_prompt: Union[List, str] = None,
                       negative_prompt_2: Optional[Union[List, str]] = None,
                       original_size: Optional[Tuple[int, int]] = None,
                       crops_coords_top_left: Tuple[int, int] = (0, 0),
                       target_size: Optional[Tuple[int, int]] = None,
                       cross_attention_kwargs: Optional[Dict[str, Any]] = None,):
        original_size = original_size or (im_size, im_size)
        target_size = target_size or (im_size, im_size)
        self.batch_size = 1 if isinstance(prompt, str) else len(prompt)
        
        #  Encode input prompt
        self.num_images_per_prompt = 1
        text_encoder_lora_scale = (
            cross_attention_kwargs.get(
                "scale", None) if cross_attention_kwargs is not None else None
        )
        (
            text_embeddings,
            negative_text_embeddings,
            pooled_text_embeddings,
            negative_pooled_text_embeddings,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=self.device,
            num_images_per_prompt=self.num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            lora_scale=text_encoder_lora_scale,
        )
        (
            _,
            unconditional_text_embeddings,
            _,
            unconditional_pooled_text_embeddings,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=self.device,
            num_images_per_prompt=self.num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            negative_prompt_2=None,
            lora_scale = text_encoder_lora_scale,
        )
        # Prepare added time ids & embeddings
        add_text_embeddings = pooled_text_embeddings
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=text_embeddings.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim
        )
        
        text_embeddings = torch.cat([negative_text_embeddings, text_embeddings, unconditional_text_embeddings], dim=0)
        add_text_embeddings = torch.cat([negative_pooled_text_embeddings, add_text_embeddings, unconditional_pooled_text_embeddings], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids, add_time_ids], dim=0)
            
            
        self.text_embeddings = text_embeddings.to(self.device)
        add_text_embeddings = add_text_embeddings.to(self.device)
        add_time_ids = add_time_ids.to(self.device).repeat(
            self.batch_size * self.num_images_per_prompt, 1)
        self.added_cond_kwargs = {
            "text_embeds": add_text_embeddings, "time_ids": add_time_ids}
        self.preprocessed_SDS = True
    
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt
        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)
        image = torch.cat([image] * 3)

        return image
    
    def score_distillation_sampling_preprocessed(self,
                                                 pred_rgb: torch.Tensor,
                                                 controlnet_images,  # controlnet image or images
                                                 im_size: int,
                                                 height: Optional[int] = None,
                                                 width: Optional[int] = None,
                                                 guidance_scale: float = 100,
                                                 grad_scale: float = 1,
                                                 t_range: Union[List[float], Tuple[float]] = (
                                                     0.05, 0.95),
                                                 use_controlnet: bool = True,
                                                 control_guidance_start: Union[float,
                                                                               List[float]] = 0.0,
                                                 control_guidance_end: Union[float,
                                                                             List[float]] = 1.0,
                                                 controlnet_conditioning_scale: Union[float,
                                                                                      List[float]] = 0.5,
                                                 cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                                                 w_mode = "cumprod", #or "constant"
                                                 ):
        assert self.preprocessed_SDS, "Please call preprocess_SDS before score_distillation_sampling_preprocessed"
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(
                control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(
                control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(self.controlnet.nets) if isinstance(
                self.controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        control_guidance_start = control_guidance_start
        control_guidance_end = control_guidance_end
        use_controlnet_coeff_scale = 1.0 if use_controlnet else 0.0
        if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [
                controlnet_conditioning_scale] * len(self.controlnet.nets)
        if isinstance(controlnet_conditioning_scale, list):
            for i, coeff in enumerate(controlnet_conditioning_scale):
                controlnet_conditioning_scale[i] = coeff * use_controlnet_coeff_scale
        else:
            controlnet_conditioning_scale = controlnet_conditioning_scale * use_controlnet_coeff_scale
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        min_step = int(num_train_timesteps * t_range[0])
        max_step = int(num_train_timesteps * t_range[1])
        alphas = self.scheduler.alphas_cumprod.to(
            self.device)  # for convenience
        self.scheduler.config.timestep_spacing = "linspace"
        self.scheduler.set_timesteps(num_train_timesteps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) <
                            s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(
                self.controlnet, ControlNetModel) else keeps)
            
        # Input augmentation
        image_list = [pred_rgb]
        if isinstance(self.controlnet, ControlNetModel):
            image_list.append(controlnet_images)
        elif isinstance(self.controlnet, MultiControlNetModel):
            image_list.extend(controlnet_images)
        image_list_a = self.x_augment_same(image_list, im_size)
        pred_rgb_a = image_list_a[0]
        if isinstance(self.controlnet, ControlNetModel):
            image_a = image_list_a[1]
        elif isinstance(self.controlnet, MultiControlNetModel):
            image_a = image_list_a[1:]
        # encode image into latents with vae, requires grad!
        latents = self.encode_(pred_rgb_a)
        latents = F.interpolate(
            latents, (128, 128), mode='bilinear', align_corners=True)
        
        
        # 4. Prepare image
        if isinstance(self.controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image_a,
                width=im_size,
                height=im_size,
                batch_size=self.batch_size * self.num_images_per_prompt,
                num_images_per_prompt=self.num_images_per_prompt,
                device=self.device,
                dtype=self.controlnet.dtype,
            )
            height, width = image.shape[-2:]
        elif isinstance(self.controlnet, MultiControlNetModel):
            images = []
            for image_ in image_a:
                image_ = self.prepare_image(
                    image=image_,
                    width=im_size,
                    height=im_size,
                    batch_size=self.batch_size * self.num_images_per_prompt,
                    num_images_per_prompt=self.num_images_per_prompt,
                    device=self.device,
                    dtype=self.controlnet.dtype,
                )
                images.append(image_)
            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False
            
        # timestep ~ U(a, b) to avoid very high/low noise level
        t = torch.randint(min_step, max_step + 1,
                          [1], dtype=torch.long, device=self.device)
        if type(self.scheduler) != diffusers.HeunDiscreteScheduler:
            while torch.sum(self.scheduler.timesteps == t).item() != 1:
                t = torch.randint(min_step, max_step + 1,
                                  [1], dtype=torch.long, device=self.device)
        # Predict the noise residual with unet
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            
            # Prediction noise
            latent_model_input = torch.cat(
                [latents_noisy] * 3)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            
            # ControlNet(s) inference
            control_model_input = latent_model_input
            controlnet_prompt_embeds = self.text_embeddings
            controlnet_added_cond_kwargs = self.added_cond_kwargs
            if isinstance(controlnet_keep[i], list):
                cond_scale = [
                    c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                conditioning_scale=cond_scale,
                guess_mode=False,
                added_cond_kwargs=controlnet_added_cond_kwargs,
                return_dict=False,
            )
            
            # Predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=self.text_embeddings,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=self.added_cond_kwargs,
                return_dict=False,
            )[0]
        
        # noise_pred_uncond is not used in Score Distillation Sampling, but can be used for other score-distillation methods
        noise_pred_neg, noise_pred_pos, noise_pred_uncond = noise_pred.chunk(3)

        if w_mode == "cumprod":
            w = (1 - alphas[t])
        elif w_mode == "constant":
            w = 1

        # Perform Classifier-Free Guidance
        grad = grad_scale * w * ((noise_pred_pos - noise) + guidance_scale * (noise_pred_pos - noise_pred_neg))
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / self.batch_size
        
        return loss, grad, pred_rgb_a, t.item()