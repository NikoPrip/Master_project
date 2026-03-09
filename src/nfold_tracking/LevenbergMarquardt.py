# -*- coding: utf-8 -*-
"""
Levenberg-Marquardt optimizer for non-linear least squares problems.

Based on the algorithm from "Numerical Recipes in C" (2002).
"""
import numpy as np
from typing import Callable, Tuple

# Make scipy optional - only needed for uncertainty estimation
try:
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except (ImportError, ValueError):
    # ValueError can occur due to NumPy version incompatibility
    SCIPY_AVAILABLE = False
    chi2 = None


class LevenbergMarquardt:
    """
    Generic Levenberg-Marquardt optimizer for non-linear least squares.

    The LM algorithm interpolates between gradient descent and Gauss-Newton:
    - High damping → gradient descent (when far from minimum)
    - Low damping → Gauss-Newton (when close to minimum)
    """

    def __init__(self,
                 function: Callable[[np.ndarray], np.ndarray],
                 initial_param: np.ndarray,
                 damping: float = 100) -> None:
        """
        Initialize the optimizer.

        Args:
            function: Residual function f(params) -> residuals (1D array)
            initial_param: Initial parameter estimate (1D array)
            damping: Initial damping factor (lambda in LM algorithm)
        """
        self.parameters_to_optimize = initial_param == initial_param  # All True
        self.damping = damping
        self.func = function
        self.param = initial_param.copy()
        self.residual_error = np.linalg.norm(self.func(self.param))
        self.coefficient_covariance_matrix = None
        self.projection_errors = None

    def jacobian(self,
                 param: np.ndarray,
                 func: Callable[[np.ndarray], np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian matrix using finite differences.

        Args:
            param: Current parameter estimate
            func: Function to compute Jacobian for

        Returns:
            (residuals, jacobian) where:
                residuals: Function values at current parameters (N,)
                jacobian: Jacobian matrix (N, M) where N=residuals, M=params
        """
        e = 0.00001  # Finite difference step size
        delta = np.zeros(param.shape)

        # Calculate the function values at the given parameters
        residuals = func(param)

        # Calculate jacobian by perturbing each parameter
        j = np.zeros((residuals.shape[0], param.shape[0]))
        for k in range(param.shape[0]):
            delta_k = delta.copy()
            delta_k[k] = e
            param_temp = param + delta_k
            func_value = func(param_temp)
            j[:, k] = (func_value - residuals) / e

        # Limit the jacobian to the parameters that should be optimized
        j = j[:, self.parameters_to_optimize]

        return (residuals, j)

    def iterate(self) -> None:
        """
        Perform one iteration of the Levenberg-Marquardt algorithm.

        Updates self.param if the step reduces the residual error.
        Adjusts damping factor based on success/failure of the step.
        """
        # Get residuals and jacobian at current parameters
        self.projection_errors, j = self.jacobian(self.param, self.func)

        # Levenberg-Marquardt update rule
        # (J^T J + lambda * diag(J^T J)) * delta = J^T * residuals
        self.coefficient_covariance_matrix = j.transpose() @ j
        damping_term = np.diag(np.diag(self.coefficient_covariance_matrix)) * self.damping
        augmented_matrix = self.coefficient_covariance_matrix + damping_term
        param_update = np.linalg.inv(augmented_matrix) @ j.transpose() @ self.projection_errors

        # Unpack to full solution (for parameters not being optimized)
        dx = np.zeros((self.param.shape[0], 1))
        dx[self.parameters_to_optimize] = param_update.reshape(-1, 1)

        # Try the update
        updated_x = self.param - dx.reshape((-1))
        updated_residual_error = np.linalg.norm(self.func(updated_x))

        if self.residual_error < updated_residual_error:
            # Squared error increased, reject update and increase damping
            self.damping = self.damping * 10
        else:
            # Squared error decreased, accept update and decrease damping
            self.param = updated_x
            self.damping = self.damping / 3
            self.residual_error = updated_residual_error

    def optimize(self, max_iterations: int = 100, tolerance: float = 1e-6) -> None:
        """
        Run optimization until convergence or max iterations.

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Stop if residual error change < tolerance
        """
        prev_error = self.residual_error
        for i in range(max_iterations):
            self.iterate()

            # Check convergence
            if abs(prev_error - self.residual_error) < tolerance:
                break
            prev_error = self.residual_error

    def estimate_uncertainties(self, p: float = 0.99) -> None:
        """
        Estimate parameter uncertainties using chi-squared statistics.

        Based on equations from "Numerical Recipes in C" (2002), Chapter 15.

        Args:
            p: Confidence level (default 0.99 for 99% confidence)
        """
        if not SCIPY_AVAILABLE:
            return

        self.squared_residual_error = self.residual_error**2

        # Determine how many standard deviations we should go out
        # to cover a given probability (p)
        self.scale_one_dim = chi2.ppf(p, 1)
        self.scale_multi_dim = chi2.ppf(p, self.param.size)

        # Equation 15.4.15 from Numerical Recipes in C 2002
        # Standard errors on parameters
        self.param_uncert = self.scale_one_dim * 1 / np.sqrt(np.diag(self.coefficient_covariance_matrix))

        # Goodness of fit test
        # Equation on page 660 in Numerical Recipes in C 2002
        self.goodness_of_fit = 1 - chi2.cdf(self.residual_error**2, self.projection_errors.size)

        # Build matrix with uncertainties for independent parameters
        delta = np.zeros(self.param.shape)
        self.independent_uncertainties = np.zeros((self.param.size, self.param.size))
        for k in range(self.param.size):
            delta_k = delta.copy()
            delta_k[k] = 1
            vector = self.param_uncert * delta_k
            self.independent_uncertainties[k, :] = vector

        # Build matrix with uncertainties for combined parameters
        # Based on equation 15.4.18 in Numerical Recipes in C 2002
        u, s, vh = np.linalg.svd(np.linalg.inv(self.coefficient_covariance_matrix))
        self.combination_uncert = self.scale_multi_dim * np.sqrt(s)
        self.combined_uncertainties = np.zeros((self.param.size, self.param.size))
        for k in range(self.param.size):
            vector = self.scale_multi_dim * vh[k] * np.sqrt(s[k])
            self.combined_uncertainties[k, :] = vector
