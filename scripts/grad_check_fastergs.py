"""Standalone finite-difference gradient check for the FasterGS rasterizer.

No scene/data needed: render a small random Gaussian cloud, define a scalar loss,
compare analytic autograd gradients to central finite differences. Run inside the
env whose FasterGSCudaBackend you want to validate (e.g. fgs_cu128).
"""
import math
import torch

from FasterGSCudaBackend.torch_bindings import diff_rasterize, RasterizerSettings

torch.manual_seed(0)
dev = "cuda"
N = 300
H = W = 64

# Random Gaussian cloud in front of the camera (view==world, camera at origin).
means = (torch.randn(N, 3, device=dev) * 0.6 + torch.tensor([0.0, 0.0, 4.0], device=dev)).requires_grad_(True)
scales = (torch.full((N, 3), math.log(0.08), device=dev) + 0.1 * torch.randn(N, 3, device=dev)).requires_grad_(True)
rot = torch.zeros(N, 4, device=dev); rot[:, 0] = 1.0
rotations = (rot + 0.01 * torch.randn(N, 4, device=dev)).requires_grad_(True)
opacities = (torch.zeros(N, 1, device=dev) + 0.5 * torch.randn(N, 1, device=dev)).requires_grad_(True)
sh0 = (0.3 * torch.randn(N, 1, 3, device=dev)).requires_grad_(True)
shr = torch.zeros(N, 15, 3, device=dev).requires_grad_(True)  # degree 0 -> unused

settings = RasterizerSettings(
    w2c=torch.eye(4, device=dev).contiguous(),
    cam_position=torch.zeros(3, device=dev).contiguous(),
    bg_color=torch.zeros(3, device=dev).contiguous(),
    active_sh_bases=1,
    width=W, height=H,
    focal_x=float(W), focal_y=float(H),
    center_x=W / 2, center_y=H / 2,
    near_plane=0.2, far_plane=100.0,
    proper_antialiasing=False,
)
target = torch.rand(3, H, W, device=dev)

def render(m, s, r, o, c0, cr):
    di = torch.zeros((2, N), dtype=torch.float32, device=dev)
    return diff_rasterize(means=m, scales=s, rotations=r, opacities=o,
                          sh_coefficients_0=c0, sh_coefficients_rest=cr,
                          densification_info=di, rasterizer_settings=settings)

def loss_of(m, s, r, o, c0, cr):
    img = render(m, s, r, o, c0, cr)
    return ((img - target) ** 2).mean()

loss = loss_of(means, scales, rotations, opacities, sh0, shr)
loss.backward()
print(f"loss={float(loss):.6f}", flush=True)

eps = 1e-3
params = {"means": means, "scales": scales, "opacities": opacities, "sh0": sh0}
for name, p in params.items():
    g = p.grad
    flat = p.detach().clone().reshape(-1)
    gflat = g.reshape(-1)
    # sample elements with the largest analytic gradient (most informative)
    idx = torch.topk(gflat.abs(), k=min(8, gflat.numel())).indices
    num, ana = [], []
    for i in idx.tolist():
        base = flat[i].item()
        fp = flat.clone(); fp[i] = base + eps
        fm = flat.clone(); fm[i] = base - eps
        with torch.no_grad():
            lp = float(loss_of(*[(fp.reshape(p.shape) if q is p else q) for q in
                                  [means, scales, rotations, opacities, sh0, shr]]))
            lm = float(loss_of(*[(fm.reshape(p.shape) if q is p else q) for q in
                                  [means, scales, rotations, opacities, sh0, shr]]))
        num.append((lp - lm) / (2 * eps)); ana.append(gflat[i].item())
    num = torch.tensor(num); ana = torch.tensor(ana)
    cos = float((num @ ana) / (num.norm() * ana.norm() + 1e-12))
    rel = float((num - ana).norm() / (ana.norm() + 1e-12))
    print(f"{name:9s} cos(num,ana)={cos:+.4f}  rel_err={rel:.3f}  ana[0:3]={[round(x,4) for x in ana[:3].tolist()]}  num[0:3]={[round(x,4) for x in num[:3].tolist()]}", flush=True)
print("GRADCHECK DONE", flush=True)
