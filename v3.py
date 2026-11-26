"""DCR, shared strong decoder, and v3 supervision."""

import math

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from rcaf import rcaf_loss


class SafeBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, self.momentum, self.eps)
        return super().forward(x)


def conv_bn_relu(cin, cout, kernel=3):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                         SafeBatchNorm(cout), nn.ReLU(inplace=True))


class StrongDecoder(nn.Module):
    def __init__(self, classes, width=128):
        super().__init__()
        self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(b),
                                               conv_bn_relu(512, width, 1))
                                  for b in (1, 2, 3, 6)])
        self.ppm_top = conv_bn_relu(512 + width * 4, width)
        self.lateral = nn.ModuleList([conv_bn_relu(c, width, 1) for c in (64, 128, 256)])
        self.smooth = nn.ModuleList([conv_bn_relu(width, width) for _ in range(4)])
        self.fuse = conv_bn_relu(width * 4, width)
        self.classifier = nn.Conv2d(width, classes, 1)
        self.deep_head = nn.Conv2d(width, classes, 1)
        self.boundary_head = nn.Conv2d(width, 1, 1)

    def forward(self, features, size):
        top = features[3]
        pooled = [F.interpolate(branch(top), size=top.shape[-2:], mode="bilinear",
                                align_corners=False) for branch in self.ppm]
        pyramid = self.ppm_top(torch.cat([top, *pooled], 1))
        levels = [None] * 4
        for i in range(3, -1, -1):
            if i < 3:
                lateral = self.lateral[i](features[i])
                pyramid = lateral + F.interpolate(pyramid, size=lateral.shape[-2:],
                                                  mode="bilinear", align_corners=False)
            levels[i] = self.smooth[i](pyramid)
            pyramid = levels[i]
        combined = self.fuse(torch.cat([F.interpolate(x, size=levels[0].shape[-2:],
                                                       mode="bilinear", align_corners=False)
                                        for x in levels], 1))
        def resize(x):
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return {"logits": resize(self.classifier(combined)),
                "deep_logits": resize(self.deep_head(levels[2])),
                "edge_logits": resize(self.boundary_head(combined))}


class DCRFusion(nn.Module):
    def __init__(self, channels, no_spatial=False, no_channel=False, no_cross=False):
        super().__init__()
        self.norm_r, self.norm_t = SafeBatchNorm(channels), SafeBatchNorm(channels)
        hidden = max(channels // 4, 1)
        self.correction = nn.Sequential(nn.Conv2d(channels * 2, hidden, 1),
                                        nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
                                        nn.GELU(), nn.Conv2d(hidden, channels, 1))
        self.channel = None if no_channel else nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels * 2, max(hidden, 16), 1),
            nn.ReLU(), nn.Conv2d(max(hidden, 16), channels, 1), nn.Sigmoid())
        self.spatial = None if no_spatial else nn.Sequential(nn.Conv2d(3, 1, 7, padding=3), nn.Sigmoid())
        self.no_cross = no_cross
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, rgb, thermal):
        rn, tn = self.norm_r(rgb), self.norm_t(thermal)
        discrepancy, product = (rn - tn).abs(), rn * tn
        pair = torch.cat((rgb, thermal), 1)
        corr = self.correction(pair if self.no_cross else torch.cat((discrepancy, product), 1))
        ac = self.channel(pair) if self.channel is not None else 1
        spatial_input = torch.cat((discrepancy.mean(1, keepdim=True),
                                   discrepancy.amax(1, keepdim=True), product.mean(1, keepdim=True)), 1)
        spatial = self.spatial(spatial_input) if self.spatial is not None else torch.ones_like(spatial_input[:, :1])
        residual = self.alpha * ac * spatial * corr
        summed = rgb + thermal
        ratio = residual.detach().float().flatten(1).norm(dim=1) / summed.detach().float().flatten(1).norm(dim=1).clamp_min(1e-8)
        diagnostics = {"alpha_abs": self.alpha.detach().abs().mean(),
                       "ratio": ratio, "spatial": spatial.detach()}
        return summed + residual, discrepancy.detach().mean(1, keepdim=True), diagnostics


def discrepancy_map(discrepancy, size):
    discrepancy = discrepancy.float()
    low = discrepancy.amin(dim=(-2, -1), keepdim=True)
    high = discrepancy.amax(dim=(-2, -1), keepdim=True)
    normal = (discrepancy - low) / (high - low).clamp_min(1e-6)
    return F.interpolate(normal, size=size, mode="bilinear", align_corners=False).detach()


def class_statistics(dataset):
    counts = torch.zeros(len(dataset.class_names), dtype=torch.int64)
    for name in dataset.names:
        with Image.open(dataset.folder / "labels" / f"{name}.png") as image:
            label = torch.from_numpy(np.array(image, dtype=np.int64))
        valid = label != dataset.ignore_index
        if ((label[valid] < 0) | (label[valid] >= len(counts))).any():
            raise ValueError(f"{name}: invalid class ids while computing class weights")
        counts += torch.bincount(label[valid], minlength=len(counts))
    if counts.sum() == 0:
        raise ValueError("training labels have no valid pixels for class weights")
    frequency = counts.double() / counts.sum()
    weights = 1 / torch.log(1.10 + frequency)
    weights /= weights.mean()
    return {"classes": dataset.class_names, "counts": counts.tolist(),
            "frequencies": frequency.tolist(), "weights": weights.tolist(),
            "ignore_index": dataset.ignore_index}


def boundary_target(target, ignore_index=255):
    valid = target != ignore_index
    padded = F.pad(target, (1, 1, 1, 1), value=ignore_index)
    edge = torch.zeros_like(valid)
    h, w = target.shape[-2:]
    for dy in range(3):
        for dx in range(3):
            neighbor = padded[:, dy:dy + h, dx:dx + w]
            edge |= valid & (neighbor != ignore_index) & (neighbor != target)
    return edge[:, None].float(), valid[:, None]


def edge_loss(logits, target, ignore_index=255):
    edge, valid = boundary_target(target, ignore_index)
    pos = (edge * valid).sum()
    neg = valid.sum() - pos
    # An edge-free batch still trains negative pixels.
    pos_weight = torch.where(pos > 0, neg / pos.clamp_min(1), pos.new_ones(())).clamp(max=20)
    losses = F.binary_cross_entropy_with_logits(logits, edge, pos_weight=pos_weight, reduction="none")
    return (losses * valid).sum() / valid.sum().clamp_min(1)


def segmentation_ce(logits, target, class_weights=None, pixel_weights=None, ignore_index=255):
    valid = target != ignore_index
    ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index, reduction="none")
    denominator = valid.sum().clamp_min(1)
    if class_weights is not None:
        denominator = class_weights[target[valid]].sum().clamp_min(1e-8)
    if pixel_weights is not None:
        mean_weight = (pixel_weights * valid).sum() / valid.sum().clamp_min(1)
        ce = ce * pixel_weights / mean_weight.clamp_min(1e-8)
    return ce.sum() / denominator


def training_loss(out, target, corrupt_label, config, legacy_weights):
    total, parts = rcaf_loss(out, target, corrupt_label, config.get("ignore_index", 255), legacy_weights)
    if (config.get("recipe", "v2") == "v2" and config.get("model") != "dcr"
            and config.get("class_weights") is None):
        return total, parts
    ignore = config.get("ignore_index", 255)
    weights = config.get("class_weights")
    weights = target.new_tensor(weights, dtype=torch.float32) if weights is not None else None
    base = segmentation_ce(out["logits"], target, weights, ignore_index=ignore)
    beta = 0 if config.get("no_dwl", False) else config.get("lambda_dwl", 1.0)
    pixels = 1 + beta * out["dwl_map"][:, 0] if "dwl_map" in out else None
    parts["main"] = segmentation_ce(out["logits"], target, weights, pixels, ignore)
    aux = [segmentation_ce(logits, F.interpolate(target[:, None].float(), size=logits.shape[-2:],
                                                 mode="nearest")[:, 0].long(), weights, ignore_index=ignore)
           for pair in out["aux"] for logits in pair]
    if aux:
        parts["aux"] = torch.stack(aux).mean()
    zero = out["logits"].sum() * 0
    parts["deep"] = (segmentation_ce(out["deep_logits"], target, weights, ignore_index=ignore)
                     if "deep_logits" in out else zero)
    parts["edge"] = edge_loss(out["edge_logits"], target, ignore) if "edge_logits" in out else zero
    parts["dwl"] = parts["main"] - base
    total = parts["main"] + sum(legacy_weights[k] * parts[k] for k in legacy_weights)
    total += config.get("lambda_deep", 0.4) * parts["deep"] + config.get("lambda_edge", 0.5) * parts["edge"]
    return total, parts


def validate_v3(config):
    for key in ("lambda_deep", "lambda_edge", "lambda_dwl"):
        value = config.get(key, {"lambda_deep": 0.4, "lambda_edge": 0.5, "lambda_dwl": 1.0}[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config.get("model") != "dcr" and (any(config.get(k, False) for k in
            ("no_dwl", "no_spatial", "no_channel", "no_cross")) or config.get("dcr_stages", "234") != "234"):
        raise ValueError("DCR ablations require --model dcr")
    stages = str(config.get("dcr_stages", "234"))
    if not stages or any(c not in "1234" for c in stages) or len(set(stages)) != len(stages):
        raise ValueError("dcr_stages must contain unique stage digits from 1 to 4")
    return sorted(int(s) for s in stages)
