"""Autodiff result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array, tree_util


@dataclass(frozen=True)
class InverseProblemResult:
    amplitude_history: Array
    gradient_history: Array
    loss_history: Array
    nu_hat: Array
    target_response: Array
    initial_response: Array
    fitted_response: Array
    sensitivity: Array
    inferred_amplitude: Array
    target_amplitude: Array


tree_util.register_dataclass(
    InverseProblemResult,
    data_fields=(
        "amplitude_history",
        "gradient_history",
        "loss_history",
        "nu_hat",
        "target_response",
        "initial_response",
        "fitted_response",
        "sensitivity",
        "inferred_amplitude",
        "target_amplitude",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class NeopaxProfileAutodiffResult:
    parameter_history: Array
    loss_history: Array
    rho: Array
    target_er_profile: Array
    fitted_er_profile: Array
    target_d11_profile: Array
    fitted_d11_profile: Array
    target_d33_profile: Array
    fitted_d33_profile: Array
    sensitivity_matrix: Array
    nu_value: Array


tree_util.register_dataclass(
    NeopaxProfileAutodiffResult,
    data_fields=(
        "parameter_history",
        "loss_history",
        "rho",
        "target_er_profile",
        "fitted_er_profile",
        "target_d11_profile",
        "fitted_d11_profile",
        "target_d33_profile",
        "fitted_d33_profile",
        "sensitivity_matrix",
        "nu_value",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class NeopaxProfileUncertaintyResult:
    rho: Array
    fitted_er_profile: Array
    fitted_d33_profile: Array
    target_d33_profile: Array
    sensitivity_matrix: Array
    parameter_covariance: Array
    parameter_std: Array
    parameter_correlation: Array
    linearized_d33_std: Array
    monte_carlo_d33_mean: Array
    monte_carlo_d33_std: Array
    monte_carlo_d33_quantile_low: Array
    monte_carlo_d33_quantile_high: Array
    sample_count: Array


tree_util.register_dataclass(
    NeopaxProfileUncertaintyResult,
    data_fields=(
        "rho",
        "fitted_er_profile",
        "fitted_d33_profile",
        "target_d33_profile",
        "sensitivity_matrix",
        "parameter_covariance",
        "parameter_std",
        "parameter_correlation",
        "linearized_d33_std",
        "monte_carlo_d33_mean",
        "monte_carlo_d33_std",
        "monte_carlo_d33_quantile_low",
        "monte_carlo_d33_quantile_high",
        "sample_count",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class BootstrapOptimizationResult:
    scale_history: Array
    gradient_history: Array
    objective_history: Array
    scale_grid: Array
    objective_landscape: Array
    rho: Array
    baseline_scale: Array
    optimized_scale: Array
    baseline_current_profile: Array
    optimized_current_profile: Array
    baseline_d13_profile: Array
    optimized_d13_profile: Array
    baseline_d33_profile: Array
    optimized_d33_profile: Array
    current_sensitivity: Array
    harmonic_m: Array
    harmonic_n: Array
    harmonic_reference_value: Array
    nu_value: Array
    serial_seconds: Array
    parallel_seconds: Array


tree_util.register_dataclass(
    BootstrapOptimizationResult,
    data_fields=(
        "scale_history",
        "gradient_history",
        "objective_history",
        "scale_grid",
        "objective_landscape",
        "rho",
        "baseline_scale",
        "optimized_scale",
        "baseline_current_profile",
        "optimized_current_profile",
        "baseline_d13_profile",
        "optimized_d13_profile",
        "baseline_d33_profile",
        "optimized_d33_profile",
        "current_sensitivity",
        "harmonic_m",
        "harmonic_n",
        "harmonic_reference_value",
        "nu_value",
        "serial_seconds",
        "parallel_seconds",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class RobustBootstrapOptimizationResult:
    scale_history: Array
    gradient_history: Array
    objective_history: Array
    scale_grid: Array
    deterministic_objective_landscape: Array
    robust_objective_landscape: Array
    rho: Array
    baseline_scale: Array
    optimized_scale: Array
    baseline_current_profile: Array
    optimized_current_profile: Array
    optimized_current_mean: Array
    optimized_current_std: Array
    optimized_current_quantile_low: Array
    optimized_current_quantile_high: Array
    harmonic_m: Array
    harmonic_n: Array
    harmonic_reference_value: Array
    nu_value: Array
    uncertainty_sigma: Array
    risk_aversion: Array


tree_util.register_dataclass(
    RobustBootstrapOptimizationResult,
    data_fields=(
        "scale_history",
        "gradient_history",
        "objective_history",
        "scale_grid",
        "deterministic_objective_landscape",
        "robust_objective_landscape",
        "rho",
        "baseline_scale",
        "optimized_scale",
        "baseline_current_profile",
        "optimized_current_profile",
        "optimized_current_mean",
        "optimized_current_std",
        "optimized_current_quantile_low",
        "optimized_current_quantile_high",
        "harmonic_m",
        "harmonic_n",
        "harmonic_reference_value",
        "nu_value",
        "uncertainty_sigma",
        "risk_aversion",
    ),
    meta_fields=(),
)


@dataclass(frozen=True)
class DerivativeAuditResult:
    nu_hat: Array
    er_hat_scan: Array
    amplitude_value: Array
    er_reference: Array
    nu_reference: Array
    autodiff_d11_da: Array
    finite_difference_d11_da: Array
    autodiff_d33_da: Array
    finite_difference_d33_da: Array
    autodiff_d11_der: Array
    finite_difference_d11_der: Array
    autodiff_d33_der: Array
    finite_difference_d33_der: Array


tree_util.register_dataclass(
    DerivativeAuditResult,
    data_fields=(
        "nu_hat",
        "er_hat_scan",
        "amplitude_value",
        "er_reference",
        "nu_reference",
        "autodiff_d11_da",
        "finite_difference_d11_da",
        "autodiff_d33_da",
        "finite_difference_d33_da",
        "autodiff_d11_der",
        "finite_difference_d11_der",
        "autodiff_d33_der",
        "finite_difference_d33_der",
    ),
    meta_fields=(),
)
