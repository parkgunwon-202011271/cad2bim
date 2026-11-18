"""조감도 실사화 작업자. 별도 파이썬 환경에서 돌아간다. 입력과 출력은 로컬 파일뿐이고 인터넷을 쓰지 않는다.

사용: python photo_worker.py 재료.npz 결과.png 첫단계.png 설정.json
AI 라이브러리가 본 프로그램의 패키지와 충돌해 프로세스를 나눴다.
"""
from __future__ import annotations

import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import (  # noqa: E402
    AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image  # noqa: E402


def main() -> None:
    data = np.load(sys.argv[1])
    settings = json.loads(open(sys.argv[4], encoding="utf-8").read())
    kind = {"torch_dtype": torch.float16, "local_files_only": True}
    control = [ControlNetModel.from_pretrained(name, variant="fp16", **kind) for name in settings["control_models"]]
    vae = AutoencoderKL.from_pretrained(settings["vae"], **kind)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(settings["model"], controlnet=control, vae=vae, variant="fp16", **kind).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.vae.enable_tiling()
    generator = torch.Generator("cuda").manual_seed(int(settings["seed"]))
    first = pipe(
        prompt=settings["prompt"], negative_prompt=settings["negative"], image=Image.fromarray(data["render"]),
        control_image=[Image.fromarray(data["depth"]).convert("RGB"), Image.fromarray(data["edges"]).convert("RGB")],
        strength=settings["strength"], controlnet_conditioning_scale=settings["control_scales"],
        num_inference_steps=settings["steps"], guidance_scale=settings["guidance"], generator=generator,
    ).images[0]
    first.save(sys.argv[3])
    width, height = first.size
    size = (int(width * settings["final_scale"]) // 8 * 8, int(height * settings["final_scale"]) // 8 * 8)
    refine = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae, text_encoder=pipe.text_encoder, text_encoder_2=pipe.text_encoder_2, tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2, unet=pipe.unet, scheduler=pipe.scheduler,
    ).to("cuda")
    final = refine(
        prompt=settings["prompt"], negative_prompt=settings["negative"], image=first.resize(size, Image.LANCZOS),
        strength=settings["refine_strength"], num_inference_steps=settings["steps"], guidance_scale=settings["guidance"], generator=generator,
    ).images[0]
    final.save(sys.argv[2])


if __name__ == "__main__":
    main()
