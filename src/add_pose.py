import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))

def add_pose(graph, initial_estimate):
    odometry = gtsam.Pose2(math.sqrt(2), math.sqrt(2), math.radians(90))
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))
    initial_estimate.insert(X(4), gtsam.Pose2(4.0 + math.sqrt(2), math.sqrt(2), math.radians(90)))
    return graph, initial_estimate