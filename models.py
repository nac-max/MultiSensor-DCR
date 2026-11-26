"""Baselines and optional ablations; the original rcaf.py stays unchanged."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from rcaf import RCAFNet, RCAFusion, _warp
from v3 import DCRFusion, StrongDecoder, discrepancy_map, validate_v3


MODEL_NAMES = ["rgb_only", "thermal_only", "early_fusion", "late_fusion",
               "add_fusion", "concat_fusion", "rcaf", "dcr"]


def encoder(channels, pretrained):
    net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    if channels != 3:
        original = net.conv1
        net.conv1 = nn.Conv2d(channels, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                if channels == 1:
                    net.conv1.weight.copy_(original.weight.mean(1, keepdim=True))
                else:
                    net.conv1.weight.copy_(torch.cat((original.weight, original.weight.mean(1, keepdim=True)), 1) * 0.75)
    net.fc, net.avgpool = nn.Identity(), nn.Identity()
    return net


class Decoder(nn.Module):
    def __init__(self, classes, width=64):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in (64, 128, 256, 512)])
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
                                     for _ in range(4)])
        self.classifier = nn.Conv2d(width, classes, 1)

    def forward(self, features, size):
        pyramid = None
        for i in range(3, -1, -1):
            lateral = self.lateral[i](features[i])
            if pyramid is not None:
                lateral = lateral + F.interpolate(pyramid, size=lateral.shape[-2:],
                                                   mode="bilinear", align_corners=False)
            pyramid = self.smooth[i](lateral)
        return F.interpolate(self.classifier(pyramid), size=size, mode="bilinear", align_corners=False)


class Baseline(nn.Module):
    def __init__(self, name, classes, pretrained, decoder="legacy", config=None):
        super().__init__()
        self.name = name
        channels = 1 if name == "thermal_only" else 4 if name == "early_fusion" else 3
        self.encoder = encoder(channels, pretrained)
        decoder_type = StrongDecoder if decoder == "strong" else Decoder
        self.decoder = decoder_type(classes)
        if name in ("late_fusion", "add_fusion", "concat_fusion", "dcr"):
            self.thermal_encoder = encoder(1, pretrained)
        if name == "late_fusion":
            self.thermal_decoder = decoder_type(classes)
        if name == "concat_fusion":
            self.reduce = nn.ModuleList([nn.Conv2d(2 * c, c, 1) for c in (64, 128, 256, 512)])
        if name == "dcr":
            config = config or {}
            channels = (64, 128, 256, 512)
            self.dcr_fusions = nn.ModuleDict({str(s): DCRFusion(channels[s - 1],
                **{k: config.get(k, False) for k in ("no_spatial", "no_channel", "no_cross")})
                for s in validate_v3(config)})

    def forward(self, rgb, thermal):
        x = thermal if self.name == "thermal_only" else torch.cat((rgb, thermal), 1) if self.name == "early_fusion" else rgb
        features = RCAFNet._encode(self.encoder, x)
        diagnostics, discrepancy = [], None
        if hasattr(self, "thermal_encoder"):
            tf = RCAFNet._encode(self.thermal_encoder, thermal)
            if self.name == "add_fusion":
                features = [r + t for r, t in zip(features, tf)]
            elif self.name == "concat_fusion":
                features = [layer(torch.cat((r, t), 1)) for layer, r, t in zip(self.reduce, features, tf)]
            elif self.name == "dcr":
                fused = []
                for stage, (r, t) in enumerate(zip(features, tf), 1):
                    if str(stage) in self.dcr_fusions:
                        feature, d, info = self.dcr_fusions[str(stage)](r, t)
                        diagnostics.append({"stage": stage, **info})
                        if discrepancy is None:
                            discrepancy = d
                    else:
                        feature = r + t
                    fused.append(feature)
                features = fused
        decoded = self.decoder(features, rgb.shape[-2:])
        out = decoded if isinstance(decoded, dict) else {"logits": decoded}
        if self.name == "late_fusion":
            other = self.thermal_decoder(tf, rgb.shape[-2:])
            if isinstance(other, dict):
                out = {key: (value + other[key]) / 2 for key, value in out.items()}
            else:
                out["logits"] = (out["logits"] + other) / 2
        out.update({"aux": [], "gates": [], "flows": []})
        if discrepancy is not None:
            out.update({"dcr": diagnostics, "dwl_map": discrepancy_map(discrepancy, rgb.shape[-2:])})
        return out


class AblatedFusion(RCAFusion):
    def __init__(self, channels, classes, no_align=False, no_reliability=False, no_interaction=False,
                 max_disp=1.0, model_version=2):
        super().__init__(channels, classes, max_disp=max_disp, model_version=model_version)
        self.no_align, self.no_reliability = no_align, no_reliability
        self.no_interaction = no_interaction
        if no_reliability:
            hidden = max(channels // 4, 16)
            self.local_gate[0] = nn.Conv2d(channels * 2, hidden, 3, padding=1)
        if no_align:
            self.offset = None
        if no_interaction:
            self.interaction = None
            self.register_parameter("gamma", None)

    def forward(self, rgb, thermal, previous_flow=None):
        pair = torch.cat((rgb, thermal), 1)
        if self.no_align:
            flow = rgb.new_zeros((rgb.shape[0], 2, *rgb.shape[-2:]))
            aligned = thermal
        else:
            flow = self.predict_flow(pair, previous_flow)
            aligned = _warp(thermal, flow)
        aux = (self.rgb_aux(rgb), self.thermal_aux(aligned))
        local = [rgb, aligned]
        if not self.no_reliability:
            local.extend(self.reliability(x) for x in aux)
        gates = (self.local_gate(torch.cat(local, 1)) + self.context(pair)).softmax(1)
        fused = self.fusion_scale * (gates[:, :1] * rgb + gates[:, 1:] * aligned)
        if not self.no_interaction:
            fused = fused + self.gamma * self.interaction(rgb * aligned)
        return fused, aux, gates, flow


def build_model(config):
    validate_v3(config)
    name = config.get("model", "rcaf")
    classes, pretrained = config.get("num_classes", 9), bool(config.get("pretrained", 1))
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name}")
    recipe = config.get("recipe", "v2")
    decoder = config.get("decoder", "strong" if recipe == "v3" else "legacy")
    if recipe not in ("v2", "v3") or decoder not in ("legacy", "strong"):
        raise ValueError("unknown recipe or decoder")
    if recipe == "v3" and decoder != "strong":
        raise ValueError("recipe v3 requires the strong decoder")
    if name != "rcaf":
        return Baseline(name, classes, pretrained, decoder, config)
    alignment = {"max_disp": config.get("max_disp", 1.0), "model_version": config.get("model_version", 2)}
    model = RCAFNet(classes, pretrained=pretrained, **alignment)
    options = {key: bool(config.get(key, False)) for key in ("no_align", "no_reliability", "no_interaction")}
    if any(options.values()):
        model.fusions = nn.ModuleList([AblatedFusion(c, classes, **options, **alignment) for c in (64, 128, 256, 512)])
    if decoder == "strong":
        del model.lateral, model.smooth, model.classifier
        model.decoder = StrongDecoder(classes)
    return model
