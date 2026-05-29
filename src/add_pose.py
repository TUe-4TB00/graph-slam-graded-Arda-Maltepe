import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate):
    # Odometry: rotate 90 degrees total (45 + 45) and move 2 meters
    odometry = gtsam.Pose2(2.0, 0.0, math.radians(90))
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # Initial estimate: X(3) is at (4,0,0), after moving 2m and rotating 90 degrees
    initial_estimate.insert(X(4), gtsam.Pose2(4.0, 2.0, math.radians(90)))

    return graph, initial_estimate