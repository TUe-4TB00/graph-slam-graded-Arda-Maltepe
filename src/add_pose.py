import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))

def add_pose(graph, initial_estimate):
    # Robot rotates 45 degrees, moves 2 meters, then rotates 45 more
    # Relative motion in X(3) frame: move 2m at 45 degrees = (sqrt(2), sqrt(2)), rotate to 90 degrees total
    odometry = gtsam.Pose2(math.sqrt(2), math.sqrt(2), math.radians(90))
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # X(3) is at (4, 0, 0), so X(4) is at (4+sqrt(2), sqrt(2), 90 degrees)
    initial_estimate.insert(X(4), gtsam.Pose2(4.0 + math.sqrt(2), math.sqrt(2), math.radians(90)))

    return graph, initial_estimate