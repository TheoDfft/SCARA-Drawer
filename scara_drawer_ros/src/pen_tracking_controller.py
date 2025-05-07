#!/usr/bin/env python3

import rospy
import numpy as np
import threading
from geometry_msgs.msg import Point, Pose, Quaternion, PoseStamped
from open_manipulator_msgs.msg import KinematicsPose
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
import tf.transformations as tft

class PenTrackerController:
    def __init__(self):
        rospy.init_node('pen_tracker_controller')
        rospy.loginfo("Starting Pen Tracker Controller node...")

        # Parameters
        self.end_effector_name = "tool"
        self.control_rate_hz = rospy.get_param("~control_rate", 20.0)  # Hz
        self.max_velocity = rospy.get_param("~max_velocity", 0.1)  # m/s
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.002) # meters
        self.desired_pose_topic = rospy.get_param("~desired_pose_topic", "/aruco_single/pose")

        if self.control_rate_hz <= 0:
            rospy.logerr("Control rate must be positive.")
            return

        self.step_distance = self.max_velocity / self.control_rate_hz
        # Path time slightly longer than control period to allow for execution
        self.path_time = 1.0 / self.control_rate_hz

        rospy.loginfo(f"Control Rate: {self.control_rate_hz} Hz")
        rospy.loginfo(f"Max Velocity: {self.max_velocity} m/s")
        rospy.loginfo(f"Step Distance: {self.step_distance:.4f} m")
        rospy.loginfo(f"Path Time: {self.path_time:.4f} s")
        rospy.loginfo(f"Goal Tolerance: {self.goal_tolerance:.4f} m")
        rospy.loginfo(f"Desired Pose Topic: {self.desired_pose_topic}")


        # State Variables
        self.latest_desired_pose = None
        self.current_actual_pose = None
        self.pose_lock = threading.Lock()

        # ROS Communication
        service_name = "/goal_task_space_path_position_only"
        rospy.loginfo(f"Waiting for service {service_name}...")
        try:
            rospy.wait_for_service(service_name, timeout=10.0)
            self.goal_service_client = rospy.ServiceProxy(
                service_name, SetKinematicsPose
            )
            rospy.loginfo("Service client connected.")
        except (rospy.ServiceException, rospy.ROSException, rospy.ROSInterruptException) as e:
            rospy.logerr(f"Failed to connect to service {service_name}: {e}")
            return # Cannot operate without the service

        self.desired_pose_sub = rospy.Subscriber(
            self.desired_pose_topic, PoseStamped, self.desired_pose_callback, queue_size=1
        )
        self.actual_pose_sub = rospy.Subscriber(
            "/tool/kinematics_pose", KinematicsPose, self.actual_pose_callback, queue_size=1
        )

        #Wait for subscribers to connect
        rospy.loginfo("Waiting for subscribers to connect...")
        rospy.sleep(1.0)

        # Control Loop Timer
        self.control_timer = rospy.Timer(
            rospy.Duration(1.5 / self.control_rate_hz), self.control_loop_callback
        )

        rospy.loginfo("Pen Tracker Controller initialized.")

    def desired_pose_callback(self, msg):
        with self.pose_lock:
            self.latest_desired_pose = msg
            rospy.logdebug(f"Received desired pose: {msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}")

    def actual_pose_callback(self, msg):
        with self.pose_lock:
            self.current_actual_pose = msg
            rospy.logdebug(f"Received actual pose: {msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}")

    def control_loop_callback(self, event):
        with self.pose_lock:
            local_desired = self.latest_desired_pose
            local_actual = self.current_actual_pose

        if local_desired is None:
            rospy.logwarn_throttle(5.0, "Waiting for first desired pose...")
            return
        if local_actual is None:
            rospy.logwarn_throttle(5.0, "Waiting for first actual pose...")
            return

        # Extract Positions as numpy arrays
        current_pos = np.array([
            local_actual.pose.position.x,
            local_actual.pose.position.y,
            local_actual.pose.position.z
        ])
        desired_pos = np.array([
            local_desired.pose.position.x,
            local_desired.pose.position.y,
            local_actual.pose.position.z
        ])

        # Calculate Intermediate Target Position
        vector_to_goal = desired_pos - current_pos
        distance_to_goal = np.linalg.norm(vector_to_goal)

        rospy.loginfo_throttle(0.5, f"Distance to goal: {distance_to_goal:.4f} m")

        if distance_to_goal < self.goal_tolerance:
            rospy.loginfo_throttle(2.0, "Goal within tolerance, skipping command.")
            return # Already at goal or very close

        if distance_to_goal <= self.step_distance:
            intermediate_target_pos_vec = desired_pos
            rospy.logdebug("Step is larger than distance, going directly to goal.")
        else:
            unit_vector = vector_to_goal / distance_to_goal
            intermediate_target_pos_vec = current_pos + unit_vector * self.step_distance
            # rospy.logdebug("Calculated intermediate step.")


        # Prepare Service Request
        req = SetKinematicsPoseRequest()
        req.end_effector_name = self.end_effector_name
        req.planning_group = self.end_effector_name # Often same for position-only
        # Use intermediate position
        req.kinematics_pose.pose.position = Point(*intermediate_target_pos_vec)
        # Use the orientation from the latest desired pose
        req.kinematics_pose.pose.orientation = local_desired.pose.orientation
        req.path_time = self.path_time

        # Call Service
        try:
            rospy.loginfo_throttle(0.5, f"Sending intermediate goal: P:[{intermediate_target_pos_vec[0]:.3f}, {intermediate_target_pos_vec[1]:.3f}, {intermediate_target_pos_vec[2]:.3f}]")
            
            response = self.goal_service_client(req)
            rospy.loginfo(f"Response: {response}")
            if not response.is_planned:
                rospy.logwarn("Intermediate goal trajectory planning failed by controller.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to {self.goal_service_client.resolved_name} failed: {e}")
        except Exception as e:
            rospy.logerr(f"An unexpected error occurred during service call: {e}")


if __name__ == '__main__':
    try:
        tracker = PenTrackerController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Pen Tracker Controller shutting down.")
    except Exception as e:
        rospy.logfatal(f"Unhandled exception: {e}")
