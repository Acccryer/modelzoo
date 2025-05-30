# Stable Diffusion XL (2023)

> [Stable Diffusion XL](https://arxiv.org/abs/2307.01952)

> **Task**: Text2Image, Inpainting

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We present SDXL, a latent diffusion model for text-to-image synthesis. Compared to previous versions of Stable Diffusion, SDXL leverages a three times larger UNet backbone: The increase of model parameters is mainly due to more attention blocks and a larger cross-attention context as SDXL uses a second text encoder. We design multiple novel conditioning schemes and train SDXL on multiple aspect ratios. We also introduce a refinement model which is used to improve the visual fidelity of samples generated by SDXL using a post-hoc image-to-image technique. We demonstrate that SDXL shows drastically improved performance compared the previous versions of Stable Diffusion and achieves results competitive with those of black-box state-of-the-art image generators.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/27d4ebad-5705-4500-826f-41f425a08c0d"/>
</div>

## Pretrained models

|                               Model                                |    Task    | Dataset | Download |
| :----------------------------------------------------------------: | :--------: | :-----: | :------: |
| [stable_diffusion_xl](./stable-diffusion_xl_ddim_denoisingunet.py) | Text2Image |    -    |    -     |

We use stable diffusion xl weights. This model has several weights including vae, unet and clip.

You may download the weights from [stable-diffusion-xl](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and change the 'from_pretrained' in config to the weights dir.

## Quick Start

Running the following codes, you can get a text-generated image.

```python
from mmengine import MODELS, Config

from mmengine.registry import init_default_scope

init_default_scope('mmagic')

config = 'configs/stable_diffusion_xl/stable-diffusion_xl_ddim_denoisingunet.py'
config = Config.fromfile(config).copy()

StableDiffuser = MODELS.build(config.model)
prompt = 'A mecha robot in a favela in expressionist style'
StableDiffuser = StableDiffuser.to('sdaa')

image = StableDiffuser.infer(prompt)['samples'][0]
image.save('robot.png')
```

## Comments

Our codebase for the stable diffusion models builds heavily on [diffusers codebase](https://github.com/huggingface/diffusers) and the model weights are from [stable-diffusion-xl](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

Thanks for the efforts of the community!
