"""Reduced-order CBF utilities for the IsaacLab G1 planar task.

This module deliberately has no IsaacLab dependency, so its geometry and
closed-form filter can be tested before Isaac Sim is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlanarCBFConfig:
    alpha: float = 5.0
    robot_radius: float = 0.35
    obstacle_radius: float = 0.30
    sigma: float = 0.5
    eps: float = 1.0e-8


def circular_barrier(
    robot_xy: torch.Tensor,
    obstacle_xy: torch.Tensor,
    robot_radius: float,
    obstacle_radius: float | torch.Tensor,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return h and grad(h) for circular obstacles.

    Shapes are ``robot_xy=(N, 2)`` and either ``obstacle_xy=(N, 2)`` or
    ``(N, M, 2)``. With multiple obstacles, the minimum-margin obstacle is
    selected independently in each environment.
    """
    if robot_xy.ndim != 2 or robot_xy.shape[-1] != 2:
        raise ValueError("robot_xy must have shape (N, 2)")
    if obstacle_xy.ndim == 2:
        obstacle_xy = obstacle_xy.unsqueeze(1)
    if obstacle_xy.ndim != 3 or obstacle_xy.shape[0] != robot_xy.shape[0] or obstacle_xy.shape[-1] != 2:
        raise ValueError("obstacle_xy must have shape (N, 2) or (N, M, 2)")

    delta = robot_xy.unsqueeze(1) - obstacle_xy
    distance = torch.linalg.vector_norm(delta, dim=-1)
    radius = torch.as_tensor(obstacle_radius, dtype=robot_xy.dtype, device=robot_xy.device)
    margin = distance - (robot_radius + radius)
    h, active = margin.min(dim=1)
    batch = torch.arange(robot_xy.shape[0], device=robot_xy.device)
    active_delta = delta[batch, active]
    active_distance = distance[batch, active].clamp_min(eps)
    grad_h = active_delta / active_distance.unsqueeze(-1)
    return h, grad_h


def closed_form_filter(
    desired_velocity: torch.Tensor,
    h: torch.Tensor,
    grad_h: torch.Tensor,
    alpha: float,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project planar velocities onto ``grad(h)^T v + alpha*h >= 0``."""
    if desired_velocity.shape != grad_h.shape or desired_velocity.shape[-1] != 2:
        raise ValueError("desired_velocity and grad_h must both have shape (N, 2)")
    if h.shape != desired_velocity.shape[:-1]:
        raise ValueError("h must have shape (N,)")

    psi = (grad_h * desired_velocity).sum(dim=-1) + alpha * h
    denominator = (grad_h * grad_h).sum(dim=-1).clamp_min(eps)
    correction = (-psi / denominator).clamp_min(0.0).unsqueeze(-1) * grad_h
    safe_velocity = desired_velocity + correction
    active = psi < 0.0
    return safe_velocity, active, psi


def cbf_reward(
    policy_velocity: torch.Tensor,
    safe_velocity: torch.Tensor,
    psi: torch.Tensor,
    sigma: float = 0.5,
    weight: float = 1.0,
) -> torch.Tensor:
    """Paper Eq. (21)/(26) CBF reward before the outer experiment weight."""
    correction_sq = ((policy_velocity - safe_velocity) ** 2).sum(dim=-1)
    return weight * (torch.minimum(psi, torch.zeros_like(psi)) + torch.exp(-correction_sq / sigma**2) - 1.0)

