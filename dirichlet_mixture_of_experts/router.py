import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Uniform, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int) -> None:
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts * 2)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(x)
        alpha_hi, alpha_lo = torch.chunk(output, 2, dim=-1)
        # enforce positivity of the alphas
        alpha_hi = torch.nn.functional.softplus(alpha_hi) + 1e-6
        alpha_lo = torch.nn.functional.softplus(alpha_lo) + 1e-6
        return alpha_hi, alpha_lo

class DirichletRouter(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_experts,
        lambda_p,
        lambda_q,
        tau_z,
        alpha_hi_prior,
        alpha_lo_prior,
        k: int = 1,
        gate_bias_init: bool = True,
        per_token_centering: bool = True,
    ):
        super(DirichletRouter, self).__init__()
        self.num_experts = num_experts
        self.encoder = Encoder(input_dim, hidden_dim, num_experts)
        self.k = k
        self.per_token_centering = per_token_centering
        self.gating_logit = nn.Linear(input_dim, num_experts, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q
        self.tau_z = tau_z
        self.logistic = TransformedDistribution(
            Uniform(0.0, 1.0),
            [SigmoidTransform().inv, AffineTransform(loc=0.0, scale=1.0)]
        )
        self.eps = 1e-6 # for numerical stability
        self.alpha_hi_prior = alpha_hi_prior
        self.alpha_lo_prior = alpha_lo_prior
        if gate_bias_init:
            self.reset_gate_bias()

    def reset_gate_bias(self) -> None:
        expected_active_fraction = min(max(self.k / self.num_experts, self.eps), 1.0 - self.eps)
        bias = self.tau_z * math.log(expected_active_fraction / (1.0 - expected_active_fraction))
        nn.init.constant_(self.gating_logit.bias, bias)

    def set_schedule(self, tau_z: float | None = None, lambda_p: float | None = None) -> None:
        if tau_z is not None:
            self.tau_z = tau_z
        if lambda_p is not None:
            self.lambda_p = lambda_p
    
    def forward(self, x):
        x_router = x.float()
        alpha_hi_post, alpha_lo_post = self.encoder(x_router)
        gating_logit = F.linear(x_router, self.gating_logit.weight, bias=None)
        if self.per_token_centering:
            gating_logit = gating_logit - gating_logit.mean(dim=-1, keepdim=True)
        gating_logit = gating_logit + self.gating_logit.bias
        noise = self.logistic.sample((x.size(0), self.num_experts)).to(device=x.device, dtype=x_router.dtype)
        z = self.sigmoid((gating_logit + noise) / self.tau_z)
        z_stopped = z.detach()
        alpha_p = self.lambda_p * (z_stopped * self.alpha_hi_prior + (1 - z_stopped) * self.alpha_lo_prior)
        alpha_p = alpha_p.clamp(min=self.eps) # for numerical stability
        alpha_q = self.lambda_q * (z * alpha_hi_post + (1 - z) * alpha_lo_post)
        alpha_q = alpha_q.clamp(min=self.eps) # for numerical stability
        dirichlet_dist = torch.distributions.Dirichlet(alpha_q)
        theta = dirichlet_dist.rsample()
        r = (z * theta) / ((z * theta).sum(dim=-1, keepdim=True) + self.eps)
        return (
            r.to(dtype=x.dtype),
            z.to(dtype=x.dtype),
            theta.to(dtype=x.dtype),
            alpha_p.to(dtype=x.dtype),
            alpha_q.to(dtype=x.dtype),
        )
