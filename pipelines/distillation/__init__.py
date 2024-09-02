from typing import AnyStr
import pathlib
from collections import OrderedDict
from packaging import version

import torch
from diffusers import StableDiffusionPipeline, SchedulerMixin, AutoencoderKL, AutoencoderTiny, ControlNetModel
from diffusers.utils import is_torch_version, is_xformers_available

huggingface_model_dict = OrderedDict({
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",  # resolution: 1024
    "ssd1b": "segmind/SSD-1B" # resolution: 1024
})

huggingface_vae_dict = OrderedDict({
    "taesdxl": ("madebyollin/taesdxl", AutoencoderTiny),
    "vae-16-fix": ("madebyollin/sdxl-vae-fp16-fix", AutoencoderKL),
})

huggingface_controlnet_dict_XL = OrderedDict({
    "canny_small": "diffusers/controlnet-canny-sdxl-1.0-small",
    "canny_mid": "diffusers/controlnet-canny-sdxl-1.0-mid",
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth_small": "diffusers/controlnet-depth-sdxl-1.0-small",
    "depth_mid": "diffusers/controlnet-depth-sdxl-1.0-mid",
    "depth": "diffusers/controlnet-depth-sdxl-1.0"
})


_model2resolution = {
    "sdxl": 1024,
    "ssd1b": 1024
}


def model2res(model_id: str):
    return _model2resolution.get(model_id, 1024)

def init_vae(model_id: str, vae_id: str):
    vae_path_ae = huggingface_vae_dict.get(vae_id, vae_id)
    if vae_path_ae is None:
        return AutoencoderKL.from_pretrained(model_id,
                                             subfolder="vae")
    else:
        vae_path, autoencoder = vae_path_ae
        return autoencoder.from_pretrained(vae_path)


def init_diffusion_pipeline(model_id: AnyStr,
                            custom_pipeline: StableDiffusionPipeline,
                            vae_id: AnyStr = None,
                            custom_scheduler: SchedulerMixin = None,
                            controlnets_id: list = None,
                            device: torch.device = "cuda",
                            torch_dtype: torch.dtype = torch.float32,
                            local_files_only: bool = True,
                            force_download: bool = False,
                            ldm_speed_up: bool = False,
                            enable_xformers: bool = True,
                            gradient_checkpoint: bool = False,
                            lora_path: AnyStr = None,
                            unet_path: AnyStr = None,
                            grayscale_prob = 0.0,
                            hflip_prob = 0.5,
                            distorsion_prob = 0.7,
                            distorsion_scale = 0.5,
                            ) -> StableDiffusionPipeline:
    """
    A tool for initial diffusers model.

    Args:
        model_id (`str` or `os.PathLike`, *optional*): pretrained_model_name_or_path
        custom_pipeline: any StableDiffusionPipeline pipeline
        custom_scheduler: any scheduler
        device: set device
        local_files_only: prohibited download model
        force_download: forced download model
        ldm_speed_up: use the `torch.compile` api to speed up unet
        enable_xformers: enable memory efficient attention from [xFormers]
        gradient_checkpoint: activates gradient checkpointing for the current model
        lora_path: load LoRA checkpoint
        unet_path: load unet checkpoint

    Returns:
            diffusers.StableDiffusionPipeline
    """

    # get model id
    model_id = huggingface_model_dict.get(model_id, model_id)
    vae = init_vae(model_id, vae_id)

    # process controlnet model
    controlnet = []
    if controlnets_id is not None:
        for controlnet_id in controlnets_id:
            controlnet_path = None     
            controlnet_path = huggingface_controlnet_dict_XL.get(controlnet_id, controlnet_id)
            if controlnet_path is None:
                raise ValueError(f"controlnet_id: {controlnet_id} is not supported.")
            controlnet_current = ControlNetModel.from_pretrained(
                controlnet_path
            ).to(device)
            controlnet.append(controlnet_current)

    # process diffusion model
    pipeline = custom_pipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        force_download=force_download,
        scheduler=custom_scheduler.from_pretrained(model_id,
                                                    subfolder="scheduler",
                                                    local_files_only=local_files_only),
        controlnet=controlnet
    ).to(device)
    
    pipeline.setup_augmentation(grayscale_prob, hflip_prob, distorsion_prob, distorsion_scale)

    if vae is not None:
        pipeline.vae = vae.to(device)
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()

    # process unet model if exist
    if unet_path is not None and pathlib.Path(unet_path).exists():
        print(f"=> load u-net from {unet_path}")
        pipeline.unet.from_pretrained(model_id, subfolder="unet")

    # process lora layers if exist
    if lora_path is not None:
        try:
            pipeline.load_lora_weights(lora_path)
            print(f"=> load lora layers into U-Net from {lora_path} ...")
        except Exception as e:
            print(f"=> error: load lora layers failed: {e}")    

    # torch.compile
    if ldm_speed_up:
        if is_torch_version(">=", "2.0.0"):
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
            print(f"=> enable torch.compile on U-Net")
        else:
            print(f"=> warning: calling torch.compile speed-up failed, since torch version <= 2.0.0")

    # Meta xformers
    if enable_xformers:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "If you observe problems during training, please update xFormers to at least 0.0.17. "
                    "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            print(f"=> enable xformers")
            pipeline.unet.enable_xformers_memory_efficient_attention()
        else:
            print(f"=> warning: calling xformers failed")

    # gradient checkpointing
    if gradient_checkpoint:
        if pipeline.unet.is_gradient_checkpointing:
            print(f"=> enable gradient checkpointing")
            pipeline.unet.enable_gradient_checkpointing()
        else:
            print("=> waring: gradient checkpointing is not activated for this model.")
    return pipeline
