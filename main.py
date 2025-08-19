# main.py
# Minimal PINN: -ε u'' + u' = 0 on [0,1], u(0)=0, u(1)=1

import os
os.environ.pop("PYCHARM_MATPLOTLIB_USE_PYCHARM_BACKEND", None)

import matplotlib
_backend_set = None
for _bk in ["TkAgg", "MacOSX", "Qt5Agg", "QtAgg"]:
    try:
        matplotlib.use(_bk, force=True)
        import matplotlib.pyplot as plt  # noqa: E402
        _ = plt.figure(); plt.close()
        _backend_set = _bk
        break
    except Exception:
        pass
if _backend_set is None:

    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

# ---------- PINN Code ----------
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# utils
def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]

def grad2(y, x):
    return grad(grad(y, x), x)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=64, depth=3):
        super().__init__()
        layers, last = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# problem
@dataclass
class Problem:
    eps: float = 1e-2
    alpha: float = 0.0   # u(0)
    beta: float  = 1.0   # u(1)

def b(x): return torch.ones_like(x)   # -ε u'' + 1*u' + 0*u = 0
def c(x): return torch.zeros_like(x)
def f(x): return torch.zeros_like(x)

# model (Dirichlet-Dirichlet, right boundary layer)
class PINN_DD(nn.Module):
    def __init__(self, prob: Problem, width=64, depth=3):
        super().__init__()
        self.eps = prob.eps
        self.alpha, self.beta = prob.alpha, prob.beta
        self.g_out = MLP(1,1,width,depth)
        self.g_in  = MLP(1,1,width,depth)
        self._lambda_raw = nn.Parameter(torch.tensor(0.0))  # λ>0 via softplus
    @property
    def lam(self): return F.softplus(self._lambda_raw) + 1e-3
    def u_out(self, x):  # left Dirichlet 硬编码
        return self.alpha + x * self.g_out(x)
    def u_in(self, x, s):  # right Dirichlet 内层修正
        x1 = torch.ones(1,1, device=x.device, dtype=x.dtype)
        u1 = self.u_out(x1)
        return torch.exp(-self.lam * s) * ((self.beta - u1) + s * self.g_in(s))
    def forward(self, x):
        x = x.requires_grad_(True)
        s = (1.0 - x) / self.eps            #  stretched
        uo = self.u_out(x)
        ui = self.u_in(x, s)
        uf = uo + ui
        # （branch-gated）
        z = torch.clamp((s - 0.5) / 1.5, 0.0, 1.0)     # s0=0.5, s1=2.0
        w_in = 1.0 - (z*z*(3 - 2*z))                   # smoothstep
        w_out = 1.0 - w_in
        w_ramp = w_in * w_out
        return uo, ui, uf, s, (w_in, w_out, w_ramp)

# physics
def L_operator(eps, x, u):
    du = grad(u, x); d2u = grad2(u, x)
    return -eps * d2u + b(x) * du + c(x) * u

def residual(prob: Problem, x, u):
    return L_operator(prob.eps, x, u) - f(x)

def loss_branch_gated(model: PINN_DD, prob: Problem, x, ramp_eta=0.1):
    uo, ui, uf, s, (w_in, w_out, w_ramp) = model(x)
    r_out = residual(prob, x, uo)
    r_in  = residual(prob, x, uf)
    loss_pde  = (w_out * r_out**2 + w_in * r_in**2).mean()
    loss_ramp = (w_ramp * r_in**2).mean()
    return loss_pde + ramp_eta * loss_ramp

# exact solution
def exact_u(eps, alpha=0.0, beta=1.0, bcoef=1.0):
    r = np.exp(-bcoef/eps)
    def u(x):
        x = np.asarray(x, dtype=np.float64)
        num = np.exp((bcoef/eps)*(x - 1.0)) - r
        den = 1.0 - r
        return alpha + (beta - alpha) * (num / den)
    return u

# training
def sample_batch(prob: Problem, n_out=800, n_in=800, s_max=6.0, device='cpu'):
    x_out = torch.rand(n_out,1, device=device)
    s = torch.rand(n_in,1, device=device) * s_max
    x_in = (1.0 - prob.eps * s).clamp(0.0, 1.0)
    x = torch.cat([x_out, x_in], 0)
    return x[torch.randperm(x.shape[0])]

def train_and_show(eps=1e-2, steps=6000, lr=2e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); np.random.seed(42)

    prob = Problem(eps=eps, alpha=0.0, beta=1.0)
    model = PINN_DD(prob, width=64, depth=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, steps+1):
        x = sample_batch(prob, device=device).requires_grad_(True)
        loss = loss_branch_gated(model, prob, x, ramp_eta=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                x0 = torch.zeros(1,1,device=device); x1 = torch.ones(1,1,device=device)
                s0 = (1-x0)/prob.eps; s1 = (1-x1)/prob.eps
                u0 = (model.u_out(x0)+model.u_in(x0,s0)).item()
                u1 = (model.u_out(x1)+model.u_in(x1,s1)).item()
            print(f"[{step:5d}] loss={loss.item():.3e}  λ={model.lam.item():.3f}  BC: u(0)={u0:.3e}, u(1)={u1:.3f}")

    # evaluate & SHOW (no saving)
    with torch.no_grad():
        xs = torch.linspace(0,1,801, device=device).view(-1,1)
        s = (1.0 - xs) / prob.eps
        u_pred = (model.u_out(xs) + model.u_in(xs, s)).cpu().numpy().ravel()
    x_np = xs.cpu().numpy().ravel()
    u_ex = exact_u(prob.eps, prob.alpha, prob.beta)(x_np)
    rel = float(np.linalg.norm(u_pred - u_ex) / np.linalg.norm(u_ex))
    print("Relative L2 error:", rel)
    if _backend_set is None:
        print("[warn] GUI backend not found (Agg is used), this environment may not be able to display images in a popup window。")

    plt.ioff()
    plt.figure(figsize=(7,4))
    plt.plot(x_np, u_pred, label="Modified PINN", linewidth=2)
    plt.plot(x_np, u_ex, "--", label="Exact", linewidth=2)
    plt.xlabel("x"); plt.ylabel("u(x)"); plt.legend()
    plt.title(f"-ε u'' + u' = 0  (ε={prob.eps})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_show(eps=1e-2, steps=6000, lr=2e-3)
