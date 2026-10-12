# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os

import cv2
import numpy as np
import torch
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru

######################################################################################
# Utility functions
######################################################################################

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    )
                                    # indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        super(EnvironmentLight, self).__init__()
        self.mtx = None      
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
        self.register_parameter('env_base', self.base)

    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        
    def build_mips(self, cutoff=0.99):
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def sample_diffuse(self, nrmvec):
        """Sample the prefiltered diffuse cubemap for world-space normals.

        P3 samples the scene-owned environment map as low-frequency
        illumination for opaque sand.  Exposing a small sampling method keeps
        nvdiffrast and cubemap orientation details inside EnvironmentLight and
        gives P4 one shared environment-light entry point.
        """

        if nrmvec.shape[-1] != 3:
            raise ValueError("environment directions must end in three channels")
        nrmvec = util.safe_normalize(nrmvec)
        if self.mtx is not None:
            mtx = torch.as_tensor(
                self.mtx, dtype=torch.float32, device=nrmvec.device
            )
            original_shape = nrmvec.shape
            nrmvec = ru.xfm_vectors(
                nrmvec.reshape(
                    original_shape[0], -1, original_shape[-1]
                ),
                mtx,
            ).reshape(*original_shape)
        return dr.texture(
            self.diffuse[None, ...],
            nrmvec.contiguous(),
            filter_mode='linear',
            boundary_mode='cube',
        )

    def sample_specular(self, reflvec, roughness):
        """Sample the roughness-prefiltered cubemap along reflection vectors.

        Inputs use channel-last layouts such as ``[1,H,W,3]`` and
        ``[1,H,W,1]``.  This is the shared low-level operation required by the
        SandyGS sand BRDF; Fresnel and material weighting deliberately remain
        outside the scene light.
        """

        if reflvec.shape[-1] != 3:
            raise ValueError("reflection directions must end in three channels")
        if roughness.shape[-1] != 1:
            raise ValueError("roughness must end in one channel")
        if reflvec.shape[:-1] != roughness.shape[:-1]:
            raise ValueError("reflection directions and roughness must align")
        reflvec = util.safe_normalize(reflvec)
        if self.mtx is not None:
            mtx = torch.as_tensor(
                self.mtx, dtype=reflvec.dtype, device=reflvec.device
            )
            original_shape = reflvec.shape
            reflvec = ru.xfm_vectors(
                reflvec.reshape(original_shape[0], -1, original_shape[-1]),
                mtx,
            ).reshape(*original_shape)
        miplevel = self.get_mip(roughness)
        return dr.texture(
            self.specular[0][None, ...],
            reflvec.contiguous(),
            mip=list(level[None, ...] for level in self.specular[1:]),
            mip_level_bias=miplevel[..., 0],
            filter_mode='linear-mipmap-linear',
            boundary_mode='cube',
        )

    def sample_fg_lut(self, ndotv, roughness):
        """Sample the preintegrated two-channel split-sum BRDF lookup."""

        if ndotv.shape[-1] != 1 or roughness.shape[-1] != 1:
            raise ValueError("NdotV and roughness must end in one channel")
        if ndotv.shape != roughness.shape:
            raise ValueError("NdotV and roughness must have identical shapes")
        fg_uv = torch.cat(
            (torch.clamp(ndotv, 1.0e-4, 1.0), torch.clamp(roughness, 0.0, 1.0)),
            dim=-1,
        )
        # Resolve relative to this module rather than the process working
        # directory; rendering scripts are routinely launched from elsewhere.
        lut_device = fg_uv.device
        if (
            not hasattr(self, '_FG_LUT')
            or self._FG_LUT.device != lut_device
            or self._FG_LUT.dtype != fg_uv.dtype
        ):
            lut_path = os.path.join(
                os.path.dirname(__file__), 'irrmaps', 'bsdf_256_256.bin'
            )
            self._FG_LUT = torch.as_tensor(
                np.fromfile(lut_path, dtype=np.float32).reshape(1, 256, 256, 2),
                dtype=fg_uv.dtype,
                device=lut_device,
            )
        return dr.texture(
            self._FG_LUT,
            fg_uv.contiguous(),
            filter_mode='linear',
            boundary_mode='clamp',
        )

    # def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
    #     # (H, W, N, 3)
    #     wo = util.safe_normalize(view_pos - gb_pos)

    #     if specular:
    #         roughness = ks[..., 1:2] # y component
    #         metallic  = ks[..., 2:3] # z component
    #         spec_col  = (1.0 - metallic)*0.04 + kd * metallic
    #         diff_col  = kd * (1.0 - metallic)
    #     else:
    #         diff_col = kd

    #     reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
    #     nrmvec = gb_normal
    #     if self.mtx is not None: # Rotate lookup
    #         mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
    #         reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
    #         nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

    #     diffuse = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
    #     shaded_col = diffuse * diff_col
    #     extras = {"diffuse": diffuse * diff_col}

    #     if specular:
    #         # Lookup FG term from lookup texture
    #         NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
    #         fg_uv = torch.cat((NdotV, roughness), dim=-1)
    #         if not hasattr(self, '_FG_LUT'):
    #             self._FG_LUT = torch.as_tensor(np.fromfile('data/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
    #         fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

    #         # Roughness adjusted specular env lookup
    #         miplevel = self.get_mip(roughness)
    #         spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

    #         # Compute aggregate lighting
    #         reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
    #         shaded_col += spec * reflectance
    #         extras["specular"] = spec * reflectance

    #     return shaded_col * (1.0 - ks[..., 0:1]), extras # Modulate by hemisphere visibility

    def shade(self, gb_pos, gb_normal, kd, ks, kr, view_pos, specular=True, is_render_moisture=False):
        # (H, W, N, C)
        wo = util.safe_normalize(view_pos - gb_pos) # wo [1, 1, N, 3]

        if specular:
            diffuse_raw = kd
            roughness = kr
            spec_col  = ks          # spec_col [1, 1, N, 3]
            diff_col  = 1.0 - ks
        else:
            raise NotImplementedError

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))  # reflvec [1, 1, N, 3]


        nrmvec = gb_normal
        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)

        #print("!!!!!!!!!!!!!!!",self.diffuse[None, ...].shape, nrmvec.contiguous().shape, gb_pos.shape, gb_normal.shape)
        ambient = self.sample_diffuse(nrmvec)
        # specular_linear = ambient * specular_tint
        specular_linear = ambient * diff_col

        if specular:
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')    # fg_lookup [1, 1, N, 2]

            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness)
            spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')
            # spec [1, 1, N, 3]

            # Compute aggregate lighting
            # reflectance = specular_tint * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]    # reflectance [1, 1, N, 3]

            # # print(spec, spec.min(), spec.max())
            # ## qy
            # if is_fluid_reflect_hit:    # 反射击中场景，则不从env map query
            #     #spec = torch.tensor([0, 0, 0], dtype=torch.float).cuda().repeat(spec.shape[2], 1).unsqueeze(0).unsqueeze(0)
            #     spec = fluid_reflect_hit_spec
            # ##

            specular_linear += spec * reflectance

        extras = {"specular": specular_linear,
                  "spec": spec}

        diffuse_linear = torch.sigmoid(diffuse_raw - np.log(3.0))
        extras["diffuse"] = diffuse_linear

        rgb = specular_linear + diffuse_linear

        if is_render_moisture:
            # Phong shading高光
            shininess = 32
            #dot_product = torch.clamp((wo * reflvec).sum(dim=-1), min=0.0)
            #dot_product = dot_product[:,:,:,None]
            #specular = ks * (dot_product ** alpha)  # [1, 1, N, 3]

            spec_angle = torch.sum(wo * reflvec, dim=-1, keepdim=True)
            specular = ks * torch.pow(torch.clamp(spec_angle, min=0.0), shininess)

            rgb = diffuse_linear + specular

            return rgb

        return rgb, extras

    def shade_ssr_nohit(self, reflvec, kr, specular=True):
        # (H, W, N, C)
        #wo = util.safe_normalize(view_pos - gb_pos) # wo [1, 1, N, 3]
        if specular:
            #diffuse_raw = kd
            roughness = kr
            #spec_col  = ks          # spec_col [1, 1, N, 3]
            #diff_col  = 1.0 - ks
        else:
            raise NotImplementedError
        reflvec = -util.safe_normalize(reflvec)
        #reflvec = util.safe_normalize(util.reflect(wo, gb_normal))  # reflvec [1, 1, N, 3]
        #nrmvec = gb_normal

        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            #nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)
        #print("!!!!!!!!!!!!!!!",self.diffuse[None, ...].shape, nrmvec.contiguous().shape, gb_pos.shape, gb_normal.shape)
        #ambient = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
        # specular_linear = ambient * specular_tint
        # specular_linear = ambient * diff_col
        if specular:
            # Lookup FG term from lookup texture
            #NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
            #fg_uv = torch.cat((NdotV, roughness), dim=-1)
            #if not hasattr(self, '_FG_LUT'):
            #    self._FG_LUT = torch.as_tensor(np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            #fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')    # fg_lookup [1, 1, N, 2]

            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness)
            #print("debug1",self.specular[0][None, ...].shape, reflvec.shape, reflvec.contiguous().shape)
            spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]),
                              mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')
            # spec [1, 1, N, 3]
            return spec

######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0):
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale   # [h,w,3]

    ## qy
    if latlong_img.max().item() > 1.:

        #latlong_img = latlong_img / (1.0 + latlong_img)    # reinhard tone mapping
        #latlong_img /= latlong_img.max()
        # gamma = 2.2
        # latlong_img = latlong_img ** (1 / gamma)

        # 上下翻转env map
        latlong_img = torch.flip(latlong_img, dims=[0])
        # 左右翻转env map
        latlong_img = torch.flip(latlong_img, dims=[1])

    ##
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])  # [6,512,512,3]

    l = EnvironmentLight(cubemap)
    l.build_mips()

    # debug_env_map = _vis_cubemap(cubemap, latlong_img.shape[0], latlong_img.shape[1]) * 255
    # cv2.imwrite("debug/debug_env_map.png", debug_env_map.cpu().numpy())

    return l


def _vis_cubemap(cubemap, width, height):   # return [Tensor, "H W 3"]
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / height, 1.0 - 1.0 / height, height, device=cubemap.device) * torch.pi,
        torch.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, device=cubemap.device) * torch.pi,
        indexing='ij',
    )
    sin_theta = gy.sin()
    reflvec = torch.stack((sin_theta * gx.sin(), gy.cos(), -sin_theta * gx.cos()), dim=-1)
    return dr.texture(cubemap[None], reflvec[None].contiguous(), filter_mode='linear', boundary_mode='cube')[0]

def load_env(fn, scale=1.0):
    if os.path.splitext(fn)[1].lower() == ".hdr":
        return _load_env_hdr(fn, scale)
    elif os.path.splitext(fn)[1].lower() == ".png":
        return _load_env_hdr(fn, scale)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def save_env_map(fn, light):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight):
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    util.save_image_raw(fn, color.detach().cpu().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    return EnvironmentLight(base)

def extract_env_map(light, resolution=[512, 1024]):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    color = util.cubemap_to_latlong(light.base, resolution)
    return color
