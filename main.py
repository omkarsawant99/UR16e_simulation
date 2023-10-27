from simulation import *
import numpy as np
import pinocchio as pin

# Define directories
package_dirs = "./urdf/"
urdf = package_dirs + "/urdf/eSeries_UR16e_15012020_URDF_ONLY.urdf"
pin_robot = pin.RobotWrapper.BuildFromURDF(urdf, package_dirs)

# Define parameters
num_joints = 6

# Initialize the environment
robot = RobotEnv(pin_robot)

# Start the visualizer
robot.start_visualizer()

# Show random positions in the visualizer 
q = np.random.rand(num_joints, 1)
robot.show_positions(q)

# Spatial Jacobian
J = robot.get_spatial_jacobian(q)

# End effector pose
T = robot.get_end_effector_pose(q)
