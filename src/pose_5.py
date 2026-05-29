import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

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

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = gtsam.NonlinearFactorGraph(graph)
            ie = gtsam.Values(initial_estimate)

            g, ie = add_pose(g, ie, pose_5)
            result = optimize(g, ie)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, ie)

            marginals = gtsam.Marginals(g, result)
            total = (marginals.marginalCovariance(L(1)).sum() +
                     marginals.marginalCovariance(L(2)).sum())

            if total < best_sum:
                best_sum = total
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_sum

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_sum = float('inf')

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = gtsam.NonlinearFactorGraph(graph)
            ie = gtsam.Values(initial_estimate)

            g, ie = add_pose(g, ie, pose_5)
            result = optimize(g, ie)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, ie)

            # Use graph error as the error metric
            list_of_errors = [g.error(result)]
            total = sum(list_of_errors)

            if total < best_sum:
                best_sum = total
                best_pose = pose_key
                best_landmark = landmark

    # Final run with best choice
    g = gtsam.NonlinearFactorGraph(graph)
    ie = gtsam.Values(initial_estimate)
    pose_5 = pose_options[best_pose]
    g, ie = add_pose(g, ie, pose_5)
    result = optimize(g, ie)
    g = add_landmark_measurement(g, result, pose_5, best_landmark)
    result = optimize(g, ie)

    list_of_errors = [g.error(result)]
    sum_of_errors = sum(list_of_errors)
    return best_pose, best_landmark, sum_of_errors