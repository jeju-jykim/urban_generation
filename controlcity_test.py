from PIL import Image
from osm_unet import OSMUNet
from utils import metadata_normalize, convert_binary
from controlnet import ControlNetModel as OSMControlNetModel
from pipeline_controlnet import StableDiffusionXLControlNetPipeline as ControlCityControlnetPipeline

from diffusers import UniPCMultistepScheduler
import torch

torch.backends.cudnn.deterministic = True

# 모델 경로
trained_controlnet_model_path = "/home/jaeyeon/jaeyeon_hdd/city/Controlcity/controlnet"
sdxl_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
trained_lora_model_path = "/home/jaeyeon/jaeyeon_hdd/city/Controlcity/lora"

# 1) ControlNet: 실가중치 로드
controlnet = OSMControlNetModel.from_pretrained(
    trained_controlnet_model_path,
    torch_dtype=torch.float32, use_safetensors=True,
    low_cpu_mem_usage=False, device_map=None
)

osm_unet = OSMUNet.from_pretrained(
    sdxl_model_path,
    subfolder="unet",
    torch_dtype=torch.float32,
    use_safetensors=True,
    low_cpu_mem_usage=False
)

pipe = ControlCityControlnetPipeline.from_pretrained(
    sdxl_model_path,
    unet=osm_unet,
    controlnet=controlnet,
    torch_dtype=torch.float32,
    use_safetensors=True,
)
pipe.unet.register_to_config(num_metadata=2)
pipe.unet.eval()
pipe.controlnet.eval()
pipe.to('cuda:0')
pipe.load_lora_weights(
    trained_lora_model_path,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


# load condition(text, metadata, cond_image, etc.)
metadata = [-122.3382568359375, 47.61727258456622]
prompt = "A black and white map of city buildings, Located in Seattle, Mostly urban area with numerous buildings, parking lots, ..."
image_road = Image.open('../examples/road/15/Seattle/5248_11443.png').convert("RGB")
image_landuse = Image.open('../examples/landuse/15/Seattle/5248_11443.png').convert("RGB")

metadata = metadata_normalize(metadata).tolist()

torch.cuda.empty_cache()
# inference
image = pipe(
    prompt=prompt,
    metadata=metadata,
    negative_prompt="Low quality.",
    image_road=image_road,
    image_landuse=image_landuse,
    guidance_scale=7.5,
    num_inference_steps=30,
    generator=torch.manual_seed(42),
    eta=0.0
).images[0]

image.save("out_raw.png")
import numpy as np
arr = np.array(image.convert("L"))   # 0~255
print("RAW min/max/mean:", arr.min(), arr.max(), arr.mean())


# image_bin_with_landuse, image_thr_with_landuse = convert_binary(image, thr=60, mode="RGB", image_landuse=image_landuse)

# image_bin2, image_thr2 = convert_binary(image, thr=60, mode="RGB")

# image_bin.save("binary_pil.png")

# import matplotlib.pyplot as plt
# plt.imshow(image_thr_with_landuse, cmap="gray", vmin=0, vmax=255)
# plt.axis("off")
# plt.tight_layout()
# plt.savefig("binary_mask.png", dpi=300, bbox_inches="tight", pad_inches=0)
# plt.close()

# plt.imshow(image_thr2, cmap="gray", vmin=0, vmax=255)
# plt.axis("off")
# plt.tight_layout()
# plt.savefig("binary_mask2.png", dpi=300, bbox_inches="tight", pad_inches=0)
# plt.close()
