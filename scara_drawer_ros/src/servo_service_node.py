#!/usr/bin/env python3

import rospy
import sys
import os
import signal
from servo_control import ServoController
from open_manipulator_msgs.srv import SetJointPosition
from open_manipulator_msgs.srv import SetJointPositionResponse

class ServoServiceNode:
    def __init__(self):
        # Initialize the node
        rospy.init_node('servo_service_node')
        rospy.loginfo("Starting Servo Service Node")
        
        # Initialize servo controller
        self.servo = ServoController(pin=17, initial_angle=0)
        rospy.loginfo("Servo controller initialized")
        
        # Create service server for gripper control
        self.tool_service = rospy.Service('/open_manipulator/goal_tool_control', 
                                         SetJointPosition, 
                                         self.tool_control_callback)
        rospy.loginfo("Tool control service created")
        
        # Set up signal handling for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        rospy.loginfo("Servo service node is ready")
    
    def tool_control_callback(self, req):
        """
        Handle tool control service requests
        Maps gripper open/close to pen_up/pen_down
        """
        position = req.joint_position.position
        
        # OpenManipulator typically uses values around -0.01 (closed) and 0.01 (open)
        # We'll use a threshold value to determine open vs closed
        if len(position) > 0:
            # Positive value typically means open gripper
            if position[0] > 0:
                rospy.loginfo("Received request to open gripper - calling pen_up()")
                self.servo.pen_up()
            # Negative or zero value typically means close gripper
            else:
                rospy.loginfo("Received request to close gripper - calling pen_down()")
                # You could map different closing values to different pen_down angles if needed
                self.servo.pen_down()
        
        # Return success response
        response = SetJointPositionResponse()
        response.is_planned = True
        return response
    
    def signal_handler(self, sig, frame):
        """Handle clean shutdown when Ctrl+C is pressed"""
        rospy.loginfo("Shutting down Servo Service Node...")
        self.servo.cleanup()
        sys.exit(0)
    
    def run(self):
        """Run the node"""
        rospy.spin()
        
if __name__ == "__main__":
    try:
        node = ServoServiceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Make sure we clean up GPIO on exit
        rospy.loginfo("Shutting down and cleaning up...")
        try:
            node.servo.cleanup()
        except:
            pass 