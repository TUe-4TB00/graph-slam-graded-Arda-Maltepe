import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))

def add_pose(graph, initial_estimate):
    # Odometry: move 2m forward, rotate 90 degrees total (45 + 45)
    # In body frame: translate sqrt(2) in x and sqrt(2) in y (diagonal), rotate 90 degrees
    odometry = gtsam.Pose2(math.sqrt(2), math.sqrt(2), math.radians(90))
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # Compute X(4) initial estimate by composing X(3) with odometry
    pose3 = initial_estimate.atPose2(X(3))
    pose4 = pose3.compose(odometry)
    initial_estimate.insert(X(4), pose4)

    return graph, initial_estimate