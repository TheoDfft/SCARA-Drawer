#!/usr/bin/env python3

import rospy
from aruco_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovariance

class MarkerPosePublisher:
    def __init__(self):
        rospy.init_node('visualize_markers', anonymous=True)
        
        # Parameters
        self.marker_size = rospy.get_param('~marker_size', 0.02)
        
        # Publishers - one for pose array and individual pose publishers
        self.pose_array_pub = rospy.Publisher('/aruco_poses', PoseArray, queue_size=10)
        self.pose_pubs = {}  # Dictionary to store publishers for individual markers
        
        # Subscribers
        rospy.Subscriber('/aruco_marker_publisher/markers', MarkerArray, self.markers_callback)
        
        rospy.loginfo("ArUco marker pose publisher initialized")
    
    def markers_callback(self, msg):
        """
        Callback for ArUco marker array - publishes poses
        """
        # Create a pose array message for all markers
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "camera_link"  # Assuming markers are in camera frame
        
        for marker in msg.markers:
            # Add pose to the pose array
            pose_array.poses.append(marker.pose.pose)
            
            # Create and publish individual PoseStamped for each marker
            marker_id = marker.id
            if marker_id not in self.pose_pubs:
                topic_name = f'/aruco_pose_{marker_id}'
                self.pose_pubs[marker_id] = rospy.Publisher(topic_name, PoseStamped, queue_size=10)
            
            # Create PoseStamped message
            pose_stamped = PoseStamped()
            pose_stamped.header = marker.header
            pose_stamped.pose = marker.pose.pose
            
            # Publish individual pose
            self.pose_pubs[marker_id].publish(pose_stamped)
            
            rospy.logdebug(f"Published pose for marker ID {marker_id}")
        
        # Publish the pose array if there are any markers
        if len(pose_array.poses) > 0:
            self.pose_array_pub.publish(pose_array)
            rospy.loginfo(f"Published poses for {len(pose_array.poses)} markers")

if __name__ == '__main__':
    try:
        publisher = MarkerPosePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 