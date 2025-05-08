#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose
from tf.transformations import quaternion_from_matrix, translation_from_matrix, \
                               concatenate_matrices, euler_from_quaternion, \
                               quaternion_from_euler, translation_matrix, quaternion_matrix

# Static transform from aruco_fused_pose (marker) to pen_tip
STATIC_TRANSFORM_TRANSLATION = [0.0, 0.032732, -0.082252]
STATIC_TRANSFORM_ROTATION = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w

pen_tip_pose_pub = None

def pose_to_matrix(pose):
    """Convert a geometry_msgs/Pose to a 4x4 transformation matrix."""
    trans = [pose.position.x, pose.position.y, pose.position.z]
    rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    return concatenate_matrices(translation_matrix(trans), quaternion_matrix(rot))

def matrix_to_pose(matrix):
    """Convert a 4x4 transformation matrix to a geometry_msgs/Pose."""
    pose = Pose()
    trans = translation_from_matrix(matrix)
    rot = quaternion_from_matrix(matrix)
    pose.position.x = trans[0]
    pose.position.y = trans[1]
    pose.position.z = trans[2]
    pose.orientation.x = rot[0]
    pose.orientation.y = rot[1]
    pose.orientation.z = rot[2]
    pose.orientation.w = rot[3]
    return pose

def aruco_pose_callback(msg):
    global pen_tip_pose_pub

    # 1. Get the pose of the marker in the world frame
    marker_world_pose = msg.pose

    # 2. Convert marker's pose to a 4x4 matrix
    marker_world_matrix = pose_to_matrix(marker_world_pose)

    # 3. Define the static transform from marker to pen_tip as a 4x4 matrix
    static_transform_trans_mat = translation_matrix(STATIC_TRANSFORM_TRANSLATION)
    static_transform_rot_mat = quaternion_matrix(STATIC_TRANSFORM_ROTATION)
    static_transform_matrix = concatenate_matrices(static_transform_trans_mat, static_transform_rot_mat)

    # 4. Calculate the pen_tip pose in the world frame
    # pen_tip_world_matrix = world_marker_matrix * marker_pen_tip_matrix
    pen_tip_world_matrix = concatenate_matrices(marker_world_matrix, static_transform_matrix)

    # 5. Convert the resulting matrix back to a Pose
    pen_tip_world_pose = matrix_to_pose(pen_tip_world_matrix)

    # 6. Create and publish the PoseStamped message for the pen_tip
    pen_tip_pose_stamped = PoseStamped()
    pen_tip_pose_stamped.header.stamp = msg.header.stamp  # Use the same timestamp
    pen_tip_pose_stamped.header.frame_id = msg.header.frame_id # Should be 'world'
    # pen_tip_pose_stamped.child_frame_id = "pen_tip" # Not standard in PoseStamped, but good for clarity
    pen_tip_pose_stamped.pose = pen_tip_world_pose

    if pen_tip_pose_pub:
        pen_tip_pose_pub.publish(pen_tip_pose_stamped)

def main():
    global pen_tip_pose_pub

    rospy.init_node('pen_tip_pose_publisher', anonymous=True)

    # Publisher for the pen_tip pose
    pen_tip_pose_pub = rospy.Publisher('/pen_tip_pose', PoseStamped, queue_size=10)

    # Subscriber to the /aruco_fused_pose topic
    rospy.Subscriber('/aruco_fused_pose', PoseStamped, aruco_pose_callback)
    rospy.sleep(1.0)

    rospy.loginfo("Pen Tip Pose Publisher node started. Subscribing to /aruco_fused_pose and publishing to /pen_tip_pose.")

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
