from typing import Tuple

import torch
from torch import nn
from torch.distributions import Uniform, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int) -> None:
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts * 2) # replace with output dim of the alphas
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(x)
        alpha_hi, alpha_lo = torch.chunk(output, 2, dim=-1)
        # enforce positivity of the alphas
        alpha_hi = torch.nn.functional.softplus(alpha_hi) + 1e-6
        alpha_lo = torch.nn.functional.softplus(alpha_lo) + 1e-6
        return alpha_hi, alpha_lo

class DirichletRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, lambda_p, lambda_q, tau_z, alpha_hi_prior, alpha_lo_prior):
        super(DirichletRouter, self).__init__()
        self.num_experts = num_experts
        self.encoder = Encoder(input_dim, hidden_dim, num_experts)
        self.gating_logit = nn.Linear(input_dim, num_experts)
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
    
    def forward(self, x):
        alpha_hi_post, alpha_lo_post = self.encoder(x)
        gating_logit = self.gating_logit(x)
        noise = self.logistic.sample((x.size(0), self.num_experts)).to(x.device)
        z = self.sigmoid((gating_logit + noise) / self.tau_z)
        z_stopped = z.detach()
        alpha_p = self.lambda_p * (z_stopped * self.alpha_hi_prior + (1 - z_stopped) * self.alpha_lo_prior)
        alpha_p = alpha_p.clamp(min=self.eps) # for numerical stability
        alpha_q = self.lambda_q * (z * alpha_hi_post + (1 - z) * alpha_lo_post)
        alpha_q = alpha_q.clamp(min=self.eps) # for numerical stability
        dirichlet_dist = torch.distributions.Dirichlet(alpha_q)
        theta = dirichlet_dist.rsample()
        r = (z * theta) / ((z * theta).sum(dim=-1, keepdim=True) + self.eps)
        return r, z, theta, alpha_p, alpha_q
