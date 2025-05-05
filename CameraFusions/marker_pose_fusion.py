#!/usr/bin/env python3
import rospy
# from geometry_msgs.msg import PoseStamped
from CameraFusions.CameraFusion import pose_fusion, Matrix3x3, Pose, Position, Quarternion

class PoseStamped:
    def __init__(self, pose: Pose):
        self.pose: Pose = pose


class MarkerPoseFusion:
    def __init__(self):
        """
        Initializes the MarkerPoseFusion node.
        """
        rospy.init_node('marker_pose_fusion_node')

        # Get topic names from parameters or use defaults
        self._input_topic_1 = rospy.get_param('~input_pose_topic_1', '/camera_left/aruco_pose')
        self._input_topic_2 = rospy.get_param('~input_pose_topic_2', '/camera_right/aruco_pose')
        self._output_topic = rospy.get_param('~output_pose_topic', '/aruco_fused_pose')

        # Initialize latest poses
        self.latest_pose_1: Pose = None
        self.latest_pose_2: Pose = None

        # Initialize publisher
        self._fused_pose_pub = rospy.Publisher(self._output_topic, PoseStamped, queue_size=10)
        #Fields for pose fusion
        self._fused_pose: Pose = Pose(Position(0, 0, 0), Quarternion(0, 0, 0, 0))
        self.camera1_cov: Matrix3x3 #TODO fill with some values - will be dynamic in the future
        self.camera2_cov: Matrix3x3 #TODO fill with some values - will be dynamic in the future

        # Initialize subscribers
        self._pose1_sub = rospy.Subscriber(self._input_topic_1, PoseStamped, self._pose1_callback)
        self._pose2_sub = rospy.Subscriber(self._input_topic_2, PoseStamped, self._pose2_callback)

        rospy.loginfo(f"Subscribing to {self._input_topic_1} and {self._input_topic_2}")
        rospy.loginfo(f"Publishing fused pose to {self._output_topic}")

    def _pose1_callback(self, msg: PoseStamped):
        """
        Callback function for the first pose topic. Updates the latest pose.
        """
        self.latest_pose_1 = msg.pose

    def _pose2_callback(self, msg: PoseStamped):
        """
        Callback function for the second pose topic. Updates the latest pose.
        """
        self.latest_pose_2 = msg.pose

    def _fuse_and_publish(self):
        """
        Fuses the latest poses and publishes the result.
        (Fusion logic to be implemented here)
        """
        # Placeholder: Check if both poses have been received
        if self.latest_pose_1 is not None and self.latest_pose_2 is not None:
            # --- Fusion Logic Goes Here ---
            self._fused_pose, _ = pose_fusion(self.latest_pose_1, self.latest_pose_2, self.camera1_cov, self.camera2_cov)
            # -----------------------------

            self._fused_pose_pub.publish(PoseStamped(self._fused_pose))
            # Reset poses after fusion to ensure we use fresh pairs? Or keep latest?
            # Decision depends on desired fusion strategy. Let's keep latest for now, otherwise:
            # self.latest_pose_1 = None 
            # self.latest_pose_2 = None
        else:
            # Optional: Log warning if trying to fuse before receiving both poses
            # rospy.logwarn_throttle(5.0, "Waiting for poses from both topics...")
            pass


    def run(self):
        """
        Main loop for the node.
        """
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self._fuse_and_publish()
            rate.sleep()

if __name__ == '__main__':
    try:
        node = MarkerPoseFusion()
        node.run()
    except rospy.ROSInterruptException:
        pass 