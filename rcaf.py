"""Reliability-Calibrated Alignment Fusion for RGB-thermal segmentation.

Offsets use feature-pixel units in (x, y) order and backward sampling.
Stage lists are returned in fine-to-coarse encoder order.
"""

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from data import RGB_MEAN, RGB_STD


def _warp(x, flow):
    b, _, h, w = x.shape
    y, x_coord = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij",
    )
    grid = torch.stack((x_coord, y), dim=-1).unsqueeze(0)
    grid = grid + flow.permute(0, 2, 3, 1)
    # align_corners=False also handles singleton spatial dimensions.
    scale = x.new_tensor((w, h))
    grid = 2 * (grid + 0.5) / scale - 1
    return F.grid_sample(x, grid.expand(b, -1, -1, -1),
                         padding_mode="border", align_corners=False)


class RCAFusion(nn.Module):
    """Per-stage alignment, detached confidence, and context-aware fusion."""

    def __init__(self, channels, num_classes, max_disp=1.0, model_version=2):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not math.isfinite(max_disp) or max_disp < 0:
            raise ValueError("max_disp must be finite and nonnegative")
        if model_version not in (1, 2):
            raise ValueError("model_version must be 1 or 2")
        self.max_disp, self.model_version = max_disp, model_version
        self.fusion_scale = 2 if model_version == 2 else 1
        hidden = max(channels // 4, 16)
        self.offset = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.rgb_aux = nn.Conv2d(channels, num_classes, 1)
        self.thermal_aux = nn.Conv2d(channels, num_classes, 1)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels * 2, hidden, 1),
            nn.ReLU(), nn.Conv2d(hidden, 2, 1),
        )
        self.local_gate = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, hidden, 3, padding=1),
            nn.ReLU(), nn.Conv2d(hidden, 2, 1),
        )
        if model_version == 2:
            for layer in (self.local_gate[-1], self.context[-1]):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.interaction = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(()))

    @staticmethod
    def reliability(logits):
        log_p = F.log_softmax(logits, dim=1)
        entropy = -(log_p.exp() * log_p).sum(dim=1, keepdim=True)
        return (1 - entropy / math.log(logits.shape[1])).clamp(0, 1).detach()

    def predict_flow(self, pair, previous_flow=None):
        raw = self.offset(pair)
        flow = self.max_disp * raw.tanh() if self.model_version == 2 else raw
        if previous_flow is not None:
            flow = flow + 2 * F.interpolate(
                previous_flow, size=pair.shape[-2:], mode="bilinear", align_corners=False,
            )
        return flow

    def forward(self, rgb, thermal, previous_flow=None):
        if rgb.shape != thermal.shape or rgb.ndim != 4:
            raise ValueError("stage features must have matching BCHW shapes")
        pair = torch.cat((rgb, thermal), dim=1)
        flow = self.predict_flow(pair, previous_flow)
        aligned = _warp(thermal, flow)
        aux = (self.rgb_aux(rgb), self.thermal_aux(aligned))
        confidence = [self.reliability(logits) for logits in aux]
        gate_logits = self.local_gate(torch.cat((rgb, aligned, *confidence), 1))
        gates = (gate_logits + self.context(pair)).softmax(dim=1)
        fused = (self.fusion_scale * (gates[:, :1] * rgb + gates[:, 1:] * aligned)
                 + self.gamma * self.interaction(rgb * aligned))
        return fused, aux, gates, flow


class RCAFNet(nn.Module):
    """Independent ResNet-18 encoders and a top-down FPN decoder."""

    def __init__(self, num_classes, pretrained=False, decoder_channels=64, max_disp=1.0, model_version=2):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.rgb_encoder = resnet18(weights=weights)
        self.thermal_encoder = resnet18(weights=weights)
        old_conv = self.thermal_encoder.conv1
        self.thermal_encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False,
        )
        if pretrained:
            with torch.no_grad():
                self.thermal_encoder.conv1.weight.copy_(old_conv.weight.mean(1, keepdim=True))
        for encoder in (self.rgb_encoder, self.thermal_encoder):
            encoder.fc = nn.Identity()
            encoder.avgpool = nn.Identity()
        channels = (64, 128, 256, 512)
        self.fusions = nn.ModuleList([RCAFusion(c, num_classes, max_disp, model_version) for c in channels])
        self.lateral = nn.ModuleList([nn.Conv2d(c, decoder_channels, 1) for c in channels])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1), nn.ReLU())
            for _ in channels
        ])
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

    @staticmethod
    def _encode(encoder, x):
        x = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(x))))
        features = []
        for layer in (encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4):
            x = layer(x)
            features.append(x)
        return features

    def forward(self, rgb, thermal):
        if (rgb.ndim != 4 or thermal.ndim != 4 or rgb.shape[1] != 3
                or thermal.shape[1] != 1 or rgb.shape[0] != thermal.shape[0]
                or rgb.shape[-2:] != thermal.shape[-2:]):
            raise ValueError("expected aligned RGB Bx3xHxW and thermal Bx1xHxW")
        rgb_features = self._encode(self.rgb_encoder, rgb)
        thermal_features = self._encode(self.thermal_encoder, thermal)
        aux, gates, flows = [None] * 4, [None] * 4, [None] * 4
        flow, pyramid = None, None
        strong_features = [None] * 4 if hasattr(self, "decoder") else None
        # Coarse-to-fine flow and decoder both run from stage 4 to stage 1.
        for i in range(3, -1, -1):
            fused, aux[i], gates[i], flow = self.fusions[i](
                rgb_features[i], thermal_features[i], flow,
            )
            flows[i] = flow
            if strong_features is not None:
                strong_features[i] = fused
                continue
            lateral = self.lateral[i](fused)
            if pyramid is not None:
                lateral = lateral + F.interpolate(
                    pyramid, size=lateral.shape[-2:], mode="bilinear", align_corners=False,
                )
            pyramid = self.smooth[i](lateral)
        if strong_features is not None:
            return {**self.decoder(strong_features, rgb.shape[-2:]),
                    "aux": aux, "gates": gates, "flows": flows}
        logits = F.interpolate(self.classifier(pyramid), size=rgb.shape[-2:],
                               mode="bilinear", align_corners=False)
        return {"logits": logits, "aux": aux, "gates": gates, "flows": flows}


def corrupt_modality(rgb, thermal, p=0.5):
    """Independently corrupt at most one modality per sample; preserve inputs."""
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    if rgb.shape[0] != thermal.shape[0] or rgb.shape[-2:] != thermal.shape[-2:]:
        raise ValueError("modalities must share batch and spatial dimensions")
    mean, std = RGB_MEAN.to(rgb), RGB_STD.to(rgb)
    outputs = [(rgb * std + mean).clone(), ((thermal + 1) / 2).clone()]
    labels = torch.full((rgb.shape[0],), -1, dtype=torch.long, device=rgb.device)
    for i in range(rgb.shape[0]):
        if torch.rand((), device=rgb.device).item() >= p:
            continue
        modality = int(torch.randint(2, (), device=rgb.device).item())
        labels[i] = modality
        operation = int(torch.randint(5, (), device=rgb.device).item())
        x = outputs[modality][i:i + 1]
        h, w = x.shape[-2:]
        if operation == 0:
            corrupted = torch.zeros_like(x)
        elif operation == 1:
            sigma = x.new_empty(()).uniform_(0.05, 0.2)
            corrupted = (x + torch.randn_like(x) * sigma).clamp(0, 1)
        elif operation == 2:
            coords = torch.arange(-2, 3, device=x.device, dtype=x.dtype)
            kernel = torch.exp(-coords.square() / 2)
            kernel = kernel[:, None] * kernel[None, :]
            kernel = (kernel / kernel.sum()).expand(x.shape[1], 1, 5, 5)
            corrupted = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"),
                                 kernel, groups=x.shape[1])
        elif operation == 3:
            eh, ew = max(1, h // 3), max(1, w // 3)
            top = int(torch.randint(h - eh + 1, ()).item())
            left = int(torch.randint(w - ew + 1, ()).item())
            corrupted = x.clone()
            corrupted[..., top:top + eh, left:left + ew] = 0
        elif modality == 0:
            corrupted = x * x.new_empty(()).uniform_(0.1, 0.5)
        else:
            flow = x.new_empty((1, 2, h, w))
            shift = torch.randint(1, max(2, min(h, w) // 10 + 1), (2,), device=x.device)
            signs = torch.randint(0, 2, (2,), device=x.device) * 2 - 1
            flow[:] = (shift * signs).view(1, 2, 1, 1)
            corrupted = _warp(x, flow)
        outputs[modality][i:i + 1] = corrupted
    corrupted_rgb = torch.where((labels == 0)[:, None, None, None], (outputs[0] - mean) / std, rgb)
    corrupted_thermal = torch.where((labels == 1)[:, None, None, None], outputs[1] * 2 - 1, thermal)
    return corrupted_rgb, corrupted_thermal, labels


def rcaf_loss(out, target, corrupt_label, ignore_index=255, lambdas=None):
    """Weighted CE, gate, TV and width-normalized flow magnitude losses."""
    weights = {"aux": 0.4, "gate": 0.1, "tv": 0.01, "mag": 0.1}
    if lambdas is not None:
        if not isinstance(lambdas, Mapping) or set(lambdas) - weights.keys():
            raise ValueError("lambdas must be a mapping with keys aux, gate, tv, mag")
        weights.update(lambdas)

    def ce(logits, labels):
        valid = labels != ignore_index
        losses = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
        return losses.sum() / valid.sum().clamp_min(1)

    main = ce(out["logits"], target)
    zero = main * 0
    aux_losses, gate_losses, tv_losses, mag_losses = [], [], [], []
    for pair in out["aux"]:
        for logits in pair:
            small_target = F.interpolate(target[:, None].float(), size=logits.shape[-2:],
                                         mode="nearest")[:, 0].long()
            aux_losses.append(ce(logits, small_target))
    selected = corrupt_label >= 0
    for gates in out["gates"]:
        if selected.any():
            means = gates.mean(dim=(-2, -1))[selected]
            corrupted_weight = means.gather(1, corrupt_label[selected, None]).squeeze(1)
            gate_losses.append(F.binary_cross_entropy(corrupted_weight, torch.zeros_like(corrupted_weight)))
    for flow in out["flows"]:
        mag_losses.append(flow.abs().mean() / flow.shape[-1])
        terms = []
        if flow.shape[-2] > 1:
            terms.append((flow[..., 1:, :] - flow[..., :-1, :]).abs().mean())
        if flow.shape[-1] > 1:
            terms.append((flow[..., 1:] - flow[..., :-1]).abs().mean())
        tv_losses.append(sum(terms, zero))

    def mean_or_zero(values):
        return torch.stack(values).mean() if values else zero

    parts = {"main": main, "aux": mean_or_zero(aux_losses),
             "gate": mean_or_zero(gate_losses), "tv": mean_or_zero(tv_losses),
             "mag": mean_or_zero(mag_losses)}
    total = main + sum(weights[key] * parts[key] for key in weights)
    return total, parts
