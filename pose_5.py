import pickle
import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_sum = float('inf')
    best_reported = float('inf')

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = pickle.loads(pickle.dumps(graph))
            est = pickle.loads(pickle.dumps(initial_estimate))

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, est)

            marginals_obj = gtsam.Marginals(g, result)
            cov_l1 = marginals_obj.marginalCovariance(L(1))
            cov_l2 = marginals_obj.marginalCovariance(L(2))
            score = np.trace(cov_l1) + np.trace(cov_l2)
            reported = cov_l1.sum() + cov_l2.sum()

            if score < best_sum:
                best_sum = score
                best_reported = reported
                best_pose = pose_name
                best_landmark = landmark

    return best_pose, best_landmark, best_reported

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_sum = float('inf')

    true_positions = [np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([4.0, 0.0])]
    pose_keys = [X(1), X(2), X(3)]

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = pickle.loads(pickle.dumps(graph))
            est = pickle.loads(pickle.dumps(initial_estimate))

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, est)

            list_of_errors = [
                float(np.linalg.norm(result.atPose2(k).translation() - t))
                for k, t in zip(pose_keys, true_positions)
            ]
            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < best_sum:
                best_sum = sum_of_errors
                best_pose = pose_name
                best_landmark = landmark

    return best_pose, best_landmark, best_sum
