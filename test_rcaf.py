"""Run with: python /home/ubuntu/rcaf/test_rcaf.py (CPU only)."""

import torch

from rcaf import RCAFNet, RCAFusion, _warp, corrupt_modality, rcaf_loss


def check_edges():
    block = RCAFusion(8, 3)
    rgb, thermal = torch.randn(2, 8, 7, 9), torch.randn(2, 8, 7, 9)
    fused, aux, gates, flow = block(rgb, thermal)
    torch.testing.assert_close(flow, torch.zeros_like(flow))
    torch.testing.assert_close(_warp(thermal, flow), thermal)
    assert not block.reliability(aux[0]).requires_grad
    previous = torch.ones(2, 2, 4, 5)
    torch.testing.assert_close(block(rgb, thermal, previous)[3], torch.full_like(flow, 2))
    torch.testing.assert_close(gates.sum(1), torch.ones_like(gates[:, 0]))
    assert fused.shape == rgb.shape

    image, heat = torch.randn(2, 3, 9, 11), torch.randn(2, 1, 9, 11)
    image_copy, heat_copy = image.clone(), heat.clone()
    clean_rgb, clean_t, labels = corrupt_modality(image, heat, p=0)
    torch.testing.assert_close(clean_rgb, image)
    torch.testing.assert_close(clean_t, heat)
    assert (labels == -1).all()
    # Both modalities have five operations; RGB dark replaces thermal shift.
    for seed in range(40):
        torch.manual_seed(seed)
        changed_rgb, changed_t, labels = corrupt_modality(image, heat, p=1)
        assert ((labels == 0) | (labels == 1)).all()
        for i, label in enumerate(labels):
            clean, original = ((changed_t, heat) if label == 0 else (changed_rgb, image))
            torch.testing.assert_close(clean[i], original[i])
        assert torch.isfinite(changed_rgb).all() and torch.isfinite(changed_t).all()
    torch.testing.assert_close(image, image_copy)
    torch.testing.assert_close(heat, heat_copy)
    for p in (-0.1, 1.1):
        try:
            corrupt_modality(image, heat, p)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid corruption probability accepted")
    try:
        RCAFusion(8, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("single-class entropy must be rejected")

    logits = torch.randn(2, 3, 1, 1, requires_grad=True)
    out = {"logits": logits, "aux": [(logits, logits)],
           "gates": [logits[:, :2].softmax(1)],
           "flows": [torch.zeros(2, 2, 1, 1, requires_grad=True)]}
    loss, parts = rcaf_loss(out, torch.full((2, 1, 1), 255), torch.full((2,), -1))
    assert loss.item() == 0 and all(value.item() == 0 for value in parts.values())
    loss.backward()
    try:
        rcaf_loss(out, torch.zeros(2, 1, 1, dtype=torch.long), torch.full((2,), -1),
                  lambdas={"unknown": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown loss weight accepted")
    print("Edge checks: passed (identity/residual flow, detached reliability, corruption, empty/error cases)")


def smoke_test():
    torch.manual_seed(7)
    model = RCAFNet(num_classes=9).cpu().train()
    rgb = torch.randn(2, 3, 480, 640)
    thermal = torch.randn(2, 1, 480, 640)
    target = torch.randint(0, 9, (2, 480, 640))
    target[:, :24, :32] = 255
    rgb, thermal, labels = corrupt_modality(rgb, thermal, p=1)
    out = model(rgb, thermal)
    assert out["logits"].shape == (2, 9, 480, 640)
    assert len(out["aux"]) == len(out["gates"]) == len(out["flows"]) == 4
    for gates in out["gates"]:
        torch.testing.assert_close(gates.sum(1), torch.ones_like(gates[:, 0]))
    loss, parts = rcaf_loss(out, target, labels)
    assert torch.isfinite(loss)
    loss.backward()
    for block in model.fusions:
        for parameter in (block.gamma, block.offset[-1].weight, block.offset[-1].bias):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    print(f"Loss: {loss.item():.6f}")
    print("Loss parts: " + ", ".join(f"{key}={value.item():.6f}" for key, value in parts.items()))
    print("Smoke test: passed (CPU, 2x480x640, forward/loss/backward, gates, gamma/offset gradients)")


if __name__ == "__main__":
    torch.set_num_threads(4)
    check_edges()
    smoke_test()
