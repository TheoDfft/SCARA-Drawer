#!/usr/bin/env python3

import rospy
import sys
import os
import signal
import time
from servo_control import ServoController
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest, SetJointPositionResponse
from open_manipulator_msgs.msg import JointPosition

class ServoServiceNode:
    def __init__(self):
        # Initialize the node
        rospy.init_node('servo_service_node')
        rospy.loginfo("Starting Servo Service Node")
        
        # Get parameters from the parameter server
        gpio_pin = rospy.get_param('~gpio_pin', 17)
        button_pin = rospy.get_param('~button_pin', 18)
        initial_angle = rospy.get_param('~initial_angle', 0)
        
        # Initialize servo controller
        try:
            self.servo = ServoController(pin=gpio_pin, button_pin=button_pin, initial_angle=initial_angle)
            rospy.loginfo(f"Servo controller initialized on GPIO pin {gpio_pin}")
        except Exception as e:
            rospy.logerr(f"Failed to initialize servo controller: {e}")
            rospy.logerr("Make sure you have permission to access GPIO pins (try running with sudo)")
            sys.exit(1)
        
        # Create service server for gripper control
        # The OpenManipulator GUI uses this service to control the gripper
        self.tool_service = rospy.Service('/goal_tool_control', 
                                         SetJointPosition, 
                                         self.tool_control_callback)
        rospy.loginfo("Tool control service created at /goal_tool_control")
        
        # Set up signal handling for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        rospy.loginfo("Servo service node is ready")
    
    def tool_control_callback(self, req):
        """
        Handle tool control service requests
        Maps gripper open/close to pen_up/pen_down
        """
        rospy.loginfo(f"Received tool control request: {req}")
        
        try:
            # Check the joint position field existence first
            if not hasattr(req, 'joint_position'):
                rospy.logerr("Request doesn't have joint_position field")
                return SetJointPositionResponse(is_planned=False)
            
            # Get the position array
            position = req.joint_position.position
            
            if len(position) > 0:
                # Positive value typically means open gripper
                if position[0] > 0:
                    rospy.loginfo("Raising pen...")
                    self.servo.pen_up()
                # Negative or zero value typically means close gripper
                else:
                    rospy.loginfo("Lowering pen...")
                    # You could map different closing values to different pen_down angles if needed
                    self.servo.pen_down()
            else:
                rospy.logwarn("Position array is empty")
                
            # Create and return a proper response
            response = SetJointPositionResponse()
            response.is_planned = True
            return response
            
        except Exception as e:
            rospy.logerr(f"Error processing tool control request: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            response = SetJointPositionResponse()
            response.is_planned = False
            return response
    
    def signal_handler(self, sig, frame):
        """Handle clean shutdown when Ctrl+C is pressed"""
        rospy.loginfo("Shutting down Servo Service Node...")
        self.servo.cleanup()
        sys.exit(0)
    
    def run(self):
        """Run the node"""
        rospy.loginfo("Servo service node running. Press Ctrl+C to exit.")

        #Continually check for the button press (active low). If pressed, call the pen_down() method.
        while not rospy.is_shutdown():
            if self.servo.button_pressed() and not self.servo.lowered_pen:
                self.servo.pen_down()
            elif not self.servo.button_pressed() and self.servo.lowered_pen:
                self.servo.pen_up()
            time.sleep(0.1)
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