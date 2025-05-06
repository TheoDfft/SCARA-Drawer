#!/usr/bin/env python3
from enum import Enum, auto
from typing import Deque
from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs import msg.Pose, msg.Position, msg.Quaternion
from CameraFusions.CameraFusion import pose_fusion, Matrix3x3, Pose, Position, Quaternion

_FILTERING_MOVING_WINDOW_LENGTH: int = 5


class FilterType(Enum):
    noFilter = auto()
    movingAverage = auto()
    orientationBased = auto()
    SLERP = auto()
    oneEuro = auto()


def _fill_pose_stamped(msg: PoseStamped, pose: Pose) -> None:
    msg.pose.Position.x = pose.Position.x
    msg.pose.Position.y = pose.Position.y
    msg.pose.Position.z = pose.Position.z

    msg.pose.Orientation.x = pose.q.x
    msg.pose.Orientation.y = pose.q.y
    msg.pose.Orientation.z = pose.q.z
    msg.pose.Orientation.w = pose.q.w

def _create_pose_from_msg_pose(msg_pose: msg.Pose) -> Pose:
    return Pose(Position(msg_pose.Position.x, msg_pose.Position.y, msg_pose.Position.z),
                Quaternion(msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y,msg.pose.orientation.z))


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
        self._fused_pose: PoseStamped = PoseStamped()

        self._pose_window: Deque[Pose] = deque(maxlen = _FILTERING_MOVING_WINDOW_LENGTH)
        '''A moving window of a number of pose measurements to filter for a smooth signal.'''
        self._filter_type: FilterType = FilterType.noFilter

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
        self.latest_pose_1 = _create_pose_from_msg_pose(msg.pose)

    def _pose2_callback(self, msg: PoseStamped):
        """
        Callback function for the second pose topic. Updates the latest pose.
        """
        self.latest_pose_2 = _create_pose_from_msg_pose(msg.pose)

    def _fuse_and_publish(self):
        """
        Fuses the latest poses and publishes the result.
        (Fusion logic to be implemented here)
        """
        #Check if both poses have been received
        if self.latest_pose_1 is not None and self.latest_pose_2 is not None:
            # --- Fusion Logic Goes Here ---
            fused_pose, _ = pose_fusion(self.latest_pose_1, self.latest_pose_2, self.camera1_cov, self.camera2_cov)
            # -----------------------------
            self._pose_window.append(fused_pose)
            _fill_pose_stamped(self._fused_pose, self.filter_pose())
            self._fused_pose_pub.publish(self._fused_pose)
            #TODO Reset poses after fusion to ensure we use fresh pairs? Or keep latest?
            # Decision depends on desired fusion strategy. Let's keep latest for now, otherwise:
            # self.latest_pose_1 = None 
            # self.latest_pose_2 = None
        else: #TODO
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

    def filter_pose(self) -> Pose:
        match self._filter_type:
            case FilterType.noFilter:
                return self._pose_window.pop()
            case FilterType.movingAverage:
                x_sum, y_sum, z_sum, qx_sum, qy_sum, qz_sum, qw_sum = 0.,0.,0.,0.,0.,0.,0.
                for pose in self._pose_window:
                    x_sum += pose.Position.x
                    y_sum += pose.Position.y
                    z_sum += pose.Position.z
                    qx_sum += pose.q.x
                    qy_sum += pose.q.y
                    qz_sum += pose.q.z
                    qw_sum += pose.q.w
                return Pose(Position(x_sum/_FILTERING_MOVING_WINDOW_LENGTH, y_sum/_FILTERING_MOVING_WINDOW_LENGTH, z_sum/_FILTERING_MOVING_WINDOW_LENGTH),
                            Quaternion(qw_sum/_FILTERING_MOVING_WINDOW_LENGTH, qx_sum/_FILTERING_MOVING_WINDOW_LENGTH, qy_sum/_FILTERING_MOVING_WINDOW_LENGTH, qz_sum/_FILTERING_MOVING_WINDOW_LENGTH))
            case FilterType.SLERP:
                pass
            case FilterType.oneEuro:
                pass
            case FilterType.orientationBased:
                pass
        
        return self._pose_window.pop()

if __name__ == '__main__':
    try:
        node = MarkerPoseFusion()
        node.run()
    except rospy.ROSInterruptException:
        pass 