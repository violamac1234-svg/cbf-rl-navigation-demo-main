import numpy as np

from arm2d.cbf import ArmCBFConfig, BarrierConstraint, build_barrier_constraints, filter_joint_velocity
from arm2d.kinematics import PlanarArm2D


def test_forward_kinematics_at_zero():
    points = PlanarArm2D().joint_positions(np.zeros(2))
    np.testing.assert_allclose(points, [[0, 0], [1, 0], [1.8, 0]], atol=1e-12)


def test_end_effector_jacobian_matches_finite_difference():
    arm = PlanarArm2D()
    q = np.array([0.4, -0.7])
    analytic = arm.end_effector_jacobian(q)
    eps = 1e-6
    numeric = np.column_stack(
        [(arm.joint_positions(q + eps * np.eye(2)[j])[-1] - arm.joint_positions(q - eps * np.eye(2)[j])[-1]) / (2 * eps)
         for j in range(2)]
    )
    np.testing.assert_allclose(analytic, numeric, atol=1e-7)


def test_two_dimensional_filter_satisfies_multiple_constraints():
    constraints = [
        BarrierConstraint(np.array([1.0, 0.0]), 0.3, 0.0, "x"),
        BarrierConstraint(np.array([0.0, 1.0]), 0.4, 0.0, "y"),
    ]
    result = filter_joint_velocity(np.array([-1.0, -1.0]), constraints, (1.5, 1.5))
    assert result.feasible
    np.testing.assert_allclose(result.velocity, [0.3, 0.4], atol=1e-8)


def test_obstacle_and_joint_limit_constraints_are_built():
    constraints = build_barrier_constraints(
        PlanarArm2D(), np.array([0.1, 0.2]), np.array([[0.8, 0.4]]), np.array([0.2]), ArmCBFConfig()
    )
    assert len(constraints) == 14  # 2 links * 5 points * 1 obstacle + 4 joint limits
    assert any(c.label == "q1_min" for c in constraints)
