import pinocchio as pin
import meshcat
import numpy as np
import scipy.linalg as sl
import time
import sys
import os
from os.path import dirname, join, abspath
 
from pinocchio.visualize import MeshcatVisualizer

# Parameters specific to UR16e
num_joints = 6

class RobotEnv:
    def __init__(self, pin_robot):
        self.model = pin_robot.model
        self.data = pin_robot.data
        self.viz = pin.visualize.MeshcatVisualizer(pin_robot.model, pin_robot.collision_model, pin_robot.visual_model)
    
    def start_visualizer(self):
        '''
        Starts a Meshcat server with visualization of the robot
        '''
        try:
            self.viz.initViewer(open=True)
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)

        # Load the robot in the viewer.
        self.viz.loadViewerModel()

        # Display a robot configuration.
        q0 = pin.neutral(self.model)
        self.viz.display(q0)
        self.viz.displayCollisions(True)
        self.viz.displayVisuals(False)
                  
    def show_positions(self, q):
        '''
        Shows the robot in the input configuration 'q'
        INPUT: q: Joint angle configuration
        '''
        self.viz.display(q)
        
    def show_neutral_positions(self):
        '''
        Shows the neutral position of the urdf file
        '''
        q0 = pin.neutral(self.model)
        print(q0)
        self.viz.display(q0)
        
    def get_spatial_jacobian(self, q):
        '''
        Returns spatial jacobian for all joints
        INPUT: q: Joint angle configuration
        OUTPUT: J = [J-linear(3 x num_joint);
                    J-angular(3 x num_joint)]
        '''
        J = pin.computeJointJacobians(self.model, self.data, q)       
        return J
    
    def get_end_effector_pose(self, q):
        '''
        Returns end effector pose
        INPUT: q: Joint angle configuration
        OUTPUT: T = Homogenous transformation
        '''
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        end_eff_id = self.model.getFrameId("Wrist3")
        end_eff = self.data.oMf[end_eff_id]
        T = np.hstack((end_eff.rotation, end_eff.translation.reshape(-1, 1)))
        T = np.vstack((T, np.array([[0, 0, 0, 1]])))
        return T
    
    def rnea(self, q, dq, ddq):
        return pin.pinocchio_pywrap.rnea(self.model, self.data, q, dq, ddq).reshape((num_joints,1))
    
    def get_gravity(self, q):
        '''
        Returns joint torques to counteract gravity
        INPUT: q: Joint angle configuration
        OUTPUT: g = Joint torques
        '''
        g = self.rnea(q, np.zeros((num_joints, 1)), np.zeros((num_joints, 1)))
        return g.reshape(-1, 1)