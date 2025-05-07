#!/usr/bin/env python

import rosbag
from geometry_msgs.msg import PoseStamped, Pose
import sys
import json
import os

def extract_poses_from_bag(bag_file_path, topic_name="/aruco_single/pose"):
    """
    Opens a rosbag, reads messages from a specified topic, and extracts poses.

    Args:
        bag_file_path (str): The path to the .bag file.
        topic_name (str): The ROS topic to extract poses from.
                          Defaults to "/tool/kinematics_pose".
    Returns:
        list: A list of geometry_msgs.msg.Pose objects.
    """
    poses = []
    try:
        print(f"Opening bag file: {bag_file_path}")
        with rosbag.Bag(bag_file_path, 'r') as bag:
            print(f"Reading messages from topic: {topic_name}")
            count = 0
            for topic, msg, t in bag.read_messages(topics=[topic_name]):
                if topic == topic_name:
                    # Check message type by comparing the _type attribute
                    is_processed = False
                    if hasattr(msg, '_type'):
                        if msg._type == PoseStamped._type:
                            poses.append(msg.pose)
                            # print(f"Extracted PoseStamped at time {t.to_sec()}")
                            count += 1
                            is_processed = True
                        elif msg._type == Pose._type:
                            poses.append(msg)
                            # print(f"Extracted Pose at time {t.to_sec()}")
                            count += 1
                            is_processed = True
                    
                    if not is_processed:
                        # Fallback or if _type attribute is missing, print a detailed warning
                        warning_msg = f"Warning: Message on topic {topic_name} is not compatible with PoseStamped or Pose."
                        if hasattr(msg, '_type'):
                            warning_msg += f" Message _type: {msg._type} (Expected PoseStamped: {PoseStamped._type} or Pose: {Pose._type})."
                        else:
                            warning_msg += " Message does not have a _type attribute."
                        warning_msg += f" Python type(msg): {type(msg)}."
                        print(warning_msg)
                        print("  Skipping.")
            if count > 0:
                print(f"Successfully extracted {count} poses from topic '{topic_name}'.")
            else:
                print(f"No poses found on topic '{topic_name}'.")
        return poses
    except rosbag.bag.BagException as e:
        print(f"Error opening or reading bag file: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during bag processing: {e}")
        return []

def save_poses_to_json(poses, output_dir, filename="extracted_tool_kinematics_poses.json"):
    """
    Saves a list of Pose objects to a JSON file.

    Args:
        poses (list): A list of geometry_msgs.msg.Pose objects.
        output_dir (str): The directory to save the JSON file in.
        filename (str): The name of the JSON file.
    """
    if not poses:
        print("No poses to save.")
        return

    serializable_poses = []
    for pose_msg in poses: # Renamed to avoid conflict with imported Pose
        serializable_pose = {
            "position": {"x": pose_msg.position.x, "y": pose_msg.position.y, "z": pose_msg.position.z},
            "orientation": {"x": pose_msg.orientation.x, "y": pose_msg.orientation.y, "z": pose_msg.orientation.z, "w": pose_msg.orientation.w}
        }
        serializable_poses.append(serializable_pose)

    try:
        if not os.path.exists(output_dir):
            print(f"Output directory {output_dir} does not exist. Creating it.")
            os.makedirs(output_dir)
        
        output_file_path = os.path.join(output_dir, filename)
        
        print(f"Saving {len(serializable_poses)} poses to {output_file_path}...")
        with open(output_file_path, 'w') as f:
            json.dump(serializable_poses, f, indent=4)
        print(f"Successfully saved poses to {output_file_path}")
    except IOError as e:
        print(f"Error writing to file {output_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during JSON serialization or file writing: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_tool_poses.py <path_to_your_rosbag.bag> [output_directory] [output_filename] [topic_name]")
        print("Example: python extract_tool_poses.py my_data.bag")
        print("Example: python extract_tool_poses.py my_data.bag ./data output_poses.json")
        print("Example: python extract_tool_poses.py my_data.bag ./data output_poses.json /my_topic")
        sys.exit(1)

    bag_file = sys.argv[1]
    
    # Determine script's directory to make relative paths for output robust
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Default output directory: one level up from 'src', then into 'data'
    # e.g. /path/to/package_name/src/script.py -> /path/to/package_name/data/
    package_dir = os.path.dirname(script_dir) 
    default_output_dir = os.path.join(package_dir, "data")

    output_directory = default_output_dir
    if len(sys.argv) > 2:
        output_directory = sys.argv[2]
        if not os.path.isabs(output_directory): # If relative, make it relative to CWD or script_dir
             output_directory = os.path.abspath(output_directory) # Make it absolute from CWD
        print(f"Custom output directory specified: {output_directory}")


    output_filename = "extracted_tool_kinematics_poses.json"
    if len(sys.argv) > 3:
        output_filename = sys.argv[3]
        print(f"Custom output filename specified: {output_filename}")

    topic_to_extract = "/aruco_single/pose" 
    if len(sys.argv) > 4:
        topic_to_extract = sys.argv[4]
        print(f"Overriding topic to extract: {topic_to_extract}")


    extracted_poses = extract_poses_from_bag(bag_file, topic_to_extract)
    
    if extracted_poses:
        save_poses_to_json(extracted_poses, output_directory, output_filename)
    else:
        print("No poses were extracted, so nothing to save.") 