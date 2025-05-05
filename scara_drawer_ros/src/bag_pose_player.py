#!/usr/bin/env python3

import rospy
import rosbag
import numpy as np
import time
import os
from open_manipulator_msgs.msg import KinematicsPose
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
from std_msgs.msg import Float64

class BagPosePlayer:
    def __init__(self):
        rospy.init_node('bag_pose_player', anonymous=True)
        
        # Parameters
        self.bag_path = rospy.get_param('~bag_file', "")
        self.min_distance_threshold = rospy.get_param('~min_distance_threshold', 0.002)  # 2 mm threshold
        self.path_time = rospy.get_param('~path_time', 0.5)  # Default path time in seconds
        
        # Resolve bag path if it contains ROS package references
        if self.bag_path.startswith('$(find'):
            try:
                pkg_name = self.bag_path.split('$(find ')[1].split(')')[0]
                pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.bag_path = self.bag_path.replace('$(find ' + pkg_name + ')', pkg_path)
                rospy.loginfo("Resolved bag path: %s", self.bag_path)
            except Exception as e:
                rospy.logerr("Failed to resolve bag path: %s", e)
        
        # Check if bag file exists and is readable
        if not os.path.exists(self.bag_path):
            rospy.logerr("Bag file does not exist: %s", self.bag_path)
            raise FileNotFoundError(f"Bag file not found: {self.bag_path}")
        
        if not os.access(self.bag_path, os.R_OK):
            rospy.logerr("Bag file is not readable: %s", self.bag_path)
            raise PermissionError(f"Cannot read bag file: {self.bag_path}")
        
        rospy.loginfo("Using bag file (verified): %s", self.bag_path)
        
        # Initialize variables
        self.current_position = None
        self.pose_received = False  # Flag to track if we've received a pose
        
        # Set up subscriber for current position
        rospy.loginfo("Setting up subscriber for current position...")
        self.current_pose_sub = rospy.Subscriber('/tool/kinematics_pose', KinematicsPose, self.current_pose_callback, queue_size=1)

        # Publisher for debugging
        self.distance_pub = rospy.Publisher('/pose_distance', Float64, queue_size=10)
        
        # Set up service client
        rospy.loginfo("Waiting for goal_task_space_path_position_only service...")
        rospy.wait_for_service('goal_task_space_path_position_only')
        self.task_space_client = rospy.ServiceProxy('goal_task_space_path_position_only', SetKinematicsPose)
        rospy.loginfo("Service found!")
    
    def current_pose_callback(self, msg):
        """Callback for receiving the current pose of the tool"""
        self.current_position = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]
        self.current_orientation = msg.pose.orientation
        self.pose_received = True
        
    def calculate_distance(self, desired_position):
        """Calculate Euclidean distance between current and desired position"""
        if self.current_position is None:
            return float('inf')  # Return infinite distance if we don't have current position
            
        position_diff = np.array(desired_position) - np.array(self.current_position)
        distance = np.linalg.norm(position_diff)
        
        # Publish distance for debugging
        debug_msg = Float64()
        debug_msg.data = distance
        self.distance_pub.publish(debug_msg)
        
        return distance
    
    def send_pose_request(self, pose_msg):
        """Send a pose request to the service"""
        req = SetKinematicsPoseRequest()
        req.end_effector_name = "tool"
        req.kinematics_pose.pose = pose_msg.pose
        req.path_time = self.path_time
        
        position = [
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z
        ]
        
        rospy.loginfo("Sending pose request to position [%.4f, %.4f, %.4f] with path_time: %.2f", 
                     position[0], position[1], position[2], req.path_time)
        
        try:
            response = self.task_space_client(req)
            if response.is_planned:
                rospy.loginfo("Pose request accepted")
                return True
            else:
                rospy.logwarn("Pose request failed planning")
                return False
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            return False
    
    def extract_poses_at_intervals(self):
        """Extract poses from bag file at regular time intervals"""
        try:
            rospy.loginfo("Opening bag file: %s", self.bag_path)
            bag = rosbag.Bag(self.bag_path)
            
            # First, find the topic with KinematicsPose messages
            rospy.loginfo("Analyzing bag contents...")
            info = bag.get_type_and_topic_info()
            rospy.loginfo("Topics in bag: %s", str(info.topics.keys()))
            
            pose_topic = None
            for topic, topic_info in info.topics.items():
                rospy.loginfo("Topic: %s, Type: %s, Count: %d", 
                             topic, topic_info.msg_type, topic_info.message_count)
                if topic_info.msg_type == 'open_manipulator_msgs/KinematicsPose':
                    pose_topic = topic
                    rospy.loginfo("Found KinematicsPose topic: %s with %d messages", 
                                 pose_topic, topic_info.message_count)
                    break
            
            if not pose_topic:
                rospy.logerr("No KinematicsPose topic found in bag file")
                bag.close()
                return []
            
            # Extract all pose messages first
            rospy.loginfo("Extracting all pose messages...")
            all_poses = []
            for topic, msg, t in bag.read_messages(topics=[pose_topic]):
                all_poses.append((t.to_sec(), msg))
            
            # Sort by timestamp
            all_poses.sort(key=lambda x: x[0])
            
            if not all_poses:
                rospy.logerr("No pose messages found in bag")
                bag.close()
                return []
            
            # Get first and last timestamp
            start_time = all_poses[0][0]
            end_time = all_poses[-1][0]
            duration = end_time - start_time
            
            rospy.loginfo("Bag timeline: start=%.2f, end=%.2f, duration=%.2f seconds", 
                         start_time, end_time, duration)
            
            # Extract poses at regular intervals
            interval_poses = []
            current_time = start_time
            
            while current_time <= end_time:
                # Find closest pose to current_time
                closest_pose = None
                min_diff = float('inf')
                
                for ts, msg in all_poses:
                    diff = abs(ts - current_time)
                    if diff < min_diff:
                        min_diff = diff
                        closest_pose = (ts, msg)
                
                if closest_pose:
                    interval_poses.append(closest_pose)
                    rospy.loginfo("Added pose at time %.2f (diff: %.3f s)", 
                                 closest_pose[0], min_diff)
                
                # Move to next interval
                current_time += self.path_time
            
            bag.close()
            rospy.loginfo("Extracted %d poses at %.2f second intervals", 
                         len(interval_poses), self.path_time)
            
            return interval_poses
            
        except Exception as e:
            rospy.logerr("Error processing bag file: %s", e)
            import traceback
            traceback.print_exc()
            return []
    
    def play_bag(self):
        """Process the bag file and send poses at intervals"""
        # Wait for current position with timeout
        start_time = rospy.Time.now()
        timeout = rospy.Duration(5.0)  # 5 seconds timeout
        
        rospy.loginfo("Waiting for current pose (timeout: 5 seconds)...")
        rate = rospy.Rate(2)  # 2Hz check rate
        
        while not rospy.is_shutdown() and not self.pose_received:
            if (rospy.Time.now() - start_time) > timeout:
                rospy.logwarn("Timeout waiting for current pose. Continuing anyway.")
                break
            
            rospy.loginfo("Still waiting for current pose... Topic: /tool/kinematics_pose")
            rate.sleep()
        
        if self.pose_received:
            rospy.loginfo("Current pose received: [%.4f, %.4f, %.4f]", 
                         self.current_position[0], 
                         self.current_position[1], 
                         self.current_position[2])
        else:
            rospy.logwarn("No current pose received, but continuing...")
        
        # Extract poses at regular intervals
        rospy.loginfo("Starting to process bag file...")
        interval_poses = self.extract_poses_at_intervals()
        
        if not interval_poses:
            rospy.logerr("No pose messages extracted at intervals")
            return
        
        # Send pose requests
        last_request_time = None
        
        rospy.loginfo("Starting to send pose requests...")
        for i, (timestamp, msg) in enumerate(interval_poses):
            if rospy.is_shutdown():
                break
            
            # Get position as a list for easier handling
            position = [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ]
            
            # Wait between requests to maintain real-time pacing
            if last_request_time is not None:
                time_since_last = (rospy.Time.now() - last_request_time).to_sec()
                if time_since_last < self.path_time:
                    sleep_time = self.path_time - time_since_last
                    rospy.loginfo("Waiting %.2f seconds before next request", sleep_time)
                    time.sleep(sleep_time)
            
            # Calculate distance to determine if we should send a request
            distance = self.calculate_distance(position)
            
            progress_pct = ((i + 1) / len(interval_poses)) * 100
            
            if distance >= self.min_distance_threshold:
                rospy.loginfo("Pose %d/%d (%.1f%%): Time=%.2f, Distance=%.4f m, sending request", 
                             i+1, len(interval_poses), progress_pct, timestamp, distance)
                
                success = self.send_pose_request(msg)
                if success:
                    last_request_time = rospy.Time.now()
                else:
                    rospy.logwarn("Request failed, continuing with next pose")
            else:
                rospy.loginfo("Pose %d/%d (%.1f%%): Time=%.2f, Distance=%.4f m, skipping (below threshold)", 
                             i+1, len(interval_poses), progress_pct, timestamp, distance)
        
        rospy.loginfo("Finished processing all poses from bag file")

def main():
    try:
        player = BagPosePlayer()
        rospy.loginfo("Starting bag pose player...")
        player.play_bag()
    except Exception as e:
        rospy.logerr("Error in main function: %s", e)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass