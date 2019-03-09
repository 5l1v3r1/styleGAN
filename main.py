"""
by @ptrblck
We appreciate your feedback. Email us at Piotr mail@pbialecki.de and Thomas tv@lernapparat.de.
Also visit https://twitter.com/ptrblck_de and https://lernapparat.de/ for more great stuff.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
    
import dnnlib, dnnlib.tflib

import collections
from collections import OrderedDict
import pickle

import numpy as np
%matplotlib inline
from matplotlib import pyplot

from .model import (G_mapping, Truncation, G_synthesis)


def main():
    g_all = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        ('truncation', Truncation(avg_latent)),
        ('g_synthesis', G_synthesis())    
    ]))

    # this can be run to get the weights, but you need the reference implementation and weights
    dnnlib.tflib.init_tf()
    weights = pickle.load(open('./karras2019stylegan-ffhq-1024x1024.pkl','rb'))
    weights_pt = [collections.OrderedDict([(k, torch.from_numpy(v.value().eval())) for k,v in w.trainables.items()]) for w in weights]
    torch.save(weights_pt, './karras2019stylegan-ffhq-1024x1024.pt')
    # then on the PyTorch side run
    state_G, state_D, state_Gs = torch.load('./karras2019stylegan-ffhq-1024x1024.pt')
    def key_translate(k):
        k = k.lower().split('/')
        if k[0] == 'g_synthesis':
            if not k[1].startswith('torgb'):
                k.insert(1, 'blocks')
            k = '.'.join(k)
            k = (k.replace('const.const','const').replace('const.bias','bias').replace('const.stylemod','epi1.style_mod.lin')
                  .replace('const.noise.weight','epi1.top_epi.noise.weight')
                  .replace('conv.noise.weight','epi2.top_epi.noise.weight')
                  .replace('conv.stylemod','epi2.style_mod.lin')
                  .replace('conv0_up.noise.weight', 'epi1.top_epi.noise.weight')
                  .replace('conv0_up.stylemod','epi1.style_mod.lin')
                  .replace('conv1.noise.weight', 'epi2.top_epi.noise.weight')
                  .replace('conv1.stylemod','epi2.style_mod.lin')
                  .replace('torgb_lod0','torgb'))
        else:
            k = '.'.join(k)
        return k

    def weight_translate(k, w):
        k = key_translate(k)
        if k.endswith('.weight'):
            if w.dim() == 2:
                w = w.t()
            elif w.dim() == 1:
                pass
            else:
                assert w.dim() == 4
                w = w.permute(3, 2, 0, 1)
        return w

    # we delete the useless torgb filters
    param_dict = {key_translate(k) : weight_translate(k, v) for k,v in state_Gs.items() if 'torgb_lod' not in key_translate(k)}
    
    sd_shapes = {k : v.shape for k,v in g_all.state_dict().items()}
    param_shapes = {k : v.shape for k,v in param_dict.items() }

    for k in list(sd_shapes)+list(param_shapes):
        pds = param_shapes.get(k)
        sds = sd_shapes.get(k)
        if pds is None:
            print ("sd only", k, sds)
        elif sds is None:
            print ("pd only", k, pds)
        elif sds != pds:
            print ("mismatch!", k, pds, sds)

    g_all.load_state_dict(param_dict, strict=False) # needed for the blur kernels
    torch.save(g_all.state_dict(), './karras2019stylegan-ffhq-1024x1024.for_g_all.pt')

    g_all.load_state_dict(torch.load('./karras2019stylegan-ffhq-1024x1024.for_g_all.pt'))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    g_all.eval()
    g_all.to(device)

    torch.manual_seed(20)
    nb_rows = 2
    nb_cols = 5
    nb_samples = nb_rows * nb_cols
    latents = torch.randn(nb_samples, 512, device=device)
    with torch.no_grad():
        imgs = g_all(latents)
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0 # normalization to 0..1 range
    imgs = imgs.cpu()

    imgs = torchvision.utils.make_grid(imgs, nrow=nb_cols)

    pyplot.figure(figsize=(15, 6))
    pyplot.imshow(imgs.permute(1, 2, 0).detach().numpy())

    # Get a few Images
    nb_latents = 25
    nb_interp = 10
    fixed_latents = [torch.randn(1, 512, device=device) for _ in range(nb_interp)]
    latents = []
    for i in range(len(fixed_latents)-1):
        latents.append(fixed_latents[i] + (fixed_latents[i + 1] - fixed_latents[i]) * torch.arange(0, 1, 0.1, device=device).unsqueeze(1))
    latents.append(fixed_latents[-1])
    latents = torch.cat(latents, dim=0)

    with torch.no_grad():
        for latent in latents:
            latent = latent.to(device)
            img = g_all(latent.unsqueeze(0))
            img = img.clamp_(-1, 1).add_(1).div_(2.0)        
            img = img.detach().squeeze(0).cpu().permute(1, 2, 0).numpy()
            # pyplot.imshow(img)
