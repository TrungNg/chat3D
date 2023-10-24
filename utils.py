import os
import torch

#PROJ_DIR = os.environ["PROJECT_DIR"]

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget, decode_latent_mesh
from shap_e.util.image_util import load_image


class Shap_e_TextTo3D:
    def __init__(self) -> None:
        self.xm = load_model('transmitter', device=_device)
        self.tmodel = load_model('text300M', device=_device)
        self.immodel = load_model('image300M', device=_device)
        self.diffusion = diffusion_from_config(load_config('diffusion'))


    def gen_from_text(self, prompt, batch_size=1, render=True, guidance_scale=10.0):
        self._prompt = prompt

        latents = sample_latents(
            batch_size=batch_size,
            model=self.tmodel,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        if render:
            self.render(latents)
        
        return latents


    def gen_from_image(self, image_path, batch_size=1, render=True, guidance_scale=3.0):
        image = load_image(image_path)

        latents = sample_latents(
            batch_size=batch_size,
            model=self.immodel,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(images=[image] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        if render:
            self.render(latents)
        
        return latents

    def render(self, latents, size=128):
        #from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
        render_mode = 'nerf' # you can change this to 'stf'
        #size = 128 # this is the size of the renders; higher values take longer to render.

        cameras = create_pan_cameras(size, _device)
        images = decode_latent_images(self.xm, latents[0], cameras, rendering_mode=render_mode)
        return gif_widget(images, 'static/sample.gif')

        #for i, latent in enumerate(latents):
        #    images = decode_latent_images(self.xm, latent, cameras, rendering_mode=render_mode)
        #    gif_widget(images, 'samples/' + self._prompt+'_'+str(i)+'.gif')


    def saveMeshes(self, latents):
        # Example of saving the latents as meshes.

        for i, latent in enumerate(latents):
            t = decode_latent_mesh(self.xm, latent).tri_mesh()
            with open(f'example_mesh_{i}.ply', 'wb') as f:
                t.write_ply(f)
            with open(f'example_mesh_{i}.obj', 'w') as f:
                t.write_obj(f)

