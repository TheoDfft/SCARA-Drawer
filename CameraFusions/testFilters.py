import json
from OneEuroFilter import OneEuroFilter
from enum import Enum, auto
from typing import Deque, Final, List
from collections import deque
import matplotlib.pyplot as plt

from CameraFusion import Pose, Position, Quaternion

_FILTERING_MOVING_WINDOW_LENGTH: Final[int] = 50


class FilterType(Enum):
    noFilter = auto()
    movingAverage = auto()
    SLERP = auto()
    oneEuro = auto()

class PoseFilter:
    # noinspection PyFinal
    def __init__(self, filter_type: FilterType):
        self.filter_type: Final[FilterType] = filter_type
        if filter_type == FilterType.noFilter:
            return
        elif filter_type == FilterType.movingAverage:
            self._pose_window: Deque[Pose] = deque(maxlen=_FILTERING_MOVING_WINDOW_LENGTH)
            '''A moving window of a number of pose measurements to filter for a smooth signal.'''
        elif filter_type == FilterType.oneEuro:
            configpx = {
                'freq': 120,  # Hz
                'mincutoff': 1.0,  # Hz
                'beta': 0.1,
                'dcutoff': 1.0
            }
            self._fpx = OneEuroFilter(**configpx)
            configpy = {
                'freq': 120,  # Hz
                'mincutoff': 1.0,  # Hz
                'beta': 0.1,
                'dcutoff': 1.0
            }
            self._fpy = OneEuroFilter(**configpy)
            configpz = {
                'freq': 120,  # Hz
                'mincutoff': 1.0,  # Hz
                'beta': 0.1,
                'dcutoff': 1.0
            }
            self._fpz = OneEuroFilter(**configpz)
            configqx = {
                'freq': 120,  # Hz
                'mincutoff': 1.0,  # Hz
                'beta': 0.1,
                'dcutoff': 1.0
            }
            self._fqx = OneEuroFilter(**configqx)
            configqy = {
                'freq': 120,  # Hz
                'mincutoff': 1.0,  # Hz
                'beta': 0.1,
                'dcutoff': 1.0
            }
            self._fqy = OneEuroFilter(**configqy)
            configqz = {
                'freq': 120,  # Hz
                'mincutoff': 1.0,  # Hz
                'beta': 0.1,
                'dcutoff': 1.0
            }
            self._fqz = OneEuroFilter(**configqz)
            configqw = {
                'freq': 120,  # Hz
                'mincutoff': 1.0,  # Hz
                'beta': 0.1,
                'dcutoff': 1.0
            }
            self._fqw = OneEuroFilter(**configqw)
        elif filter_type == FilterType.SLERP:
            # self._h = 0.1  # Normalized cutoff (0-1)
            # self._current_q = np.array([1, 0, 0, 0])
            pass
        else:
            pass

    def filter_pose(self, pose: Pose) -> Pose:
        match self.filter_type:
            case FilterType.noFilter:
                return pose
            case FilterType.movingAverage:
                self._pose_window.append(pose)
                x_sum, y_sum, z_sum, qx_sum, qy_sum, qz_sum, qw_sum = 0., 0., 0., 0., 0., 0., 0.
                for pose in self._pose_window:
                    x_sum += pose.Position.x
                    y_sum += pose.Position.y
                    z_sum += pose.Position.z
                    qx_sum += pose.q.x
                    qy_sum += pose.q.y
                    qz_sum += pose.q.z
                    qw_sum += pose.q.w
                length: int = min(_FILTERING_MOVING_WINDOW_LENGTH, len(self._pose_window))
                return Pose(Position(x_sum / length, y_sum / length, z_sum / length),
                            Quaternion(qw_sum / length, qx_sum / length, qy_sum / length, qz_sum / length))
            case FilterType.SLERP:
                # if pose.q @ self._current_q < 0:
                #     pose.q = -pose.q
                #
                #     # Adaptive interpolation factor based on angular distance[1][2]
                # theta = np.arccos(np.clip(pose.q @ self._current_q, -1, 1))
                # h_adaptive = self.h * (theta / np.pi)  # Scale h by angular difference[2]
                #
                # # Apply SLERP[4]
                # self._current_q = geometric_slerp(self._current_q, pose.q, [h_adaptive])[0]
                # pose.q = self._current_q
                # return pose
                pass
            case FilterType.oneEuro:
                return Pose(
                    Position(self._fpx(pose.Position.x), self._fpy(pose.Position.y), self._fpz(pose.Position.z)),
                    Quaternion(self._fqx(pose.q.x), self._fqy(pose.q.y), self._fqz(pose.q.z), self._fqw(pose.q.w)))
            case FilterType.orientationBased:
                pass

        return pose


def read_poses_from_json(json_path: str) -> List[Pose]:
    with open(json_path, 'r') as f:
        data =json.load(f)
    pose_list: List[Pose] = []
    for pose_dict in data:
        pose_list.append(Pose(Position(float(pose_dict["position"]["x"]),
                                       float(pose_dict["position"]["y"]),
                                       float(pose_dict["position"]["z"])),
                         Quaternion(float(pose_dict["orientation"]["w"]),
                                    float(pose_dict["orientation"]["x"]),
                                    float(pose_dict["orientation"]["y"]),
                                    float(pose_dict["orientation"]["z"]))))
    return pose_list

if __name__ == '__main__':
    import os
    current_directory = os.getcwd()
    json_path: str = os.path.join(current_directory, "extracted_tool_kinematics_poses.json")
    print(json_path)
    poses: List[Pose] = read_poses_from_json(json_path)

    pos_x_list: List[float] = [pose.Position.x for pose in poses]
    pos_y_list: List[float] = [pose.Position.y for pose in poses]
    pos_z_list: List[float] = [pose.Position.z for pose in poses]
    q_w_list: List[float] = [pose.q.w for pose in poses]
    q_x_list: List[float] = [pose.q.x for pose in poses]
    q_y_list: List[float] = [pose.q.y for pose in poses]
    q_z_list: List[float] = [pose.q.z for pose in poses]


    values: List[float] = q_y_list
    plt.plot(values)
    plt.show()

    filter_type : FilterType = FilterType.movingAverage
    pose_filter: PoseFilter = PoseFilter(filter_type)

    filtered_poses: List[Pose] = []
    for pose in poses:
        filtered_poses.append(pose_filter.filter_pose(pose))

    filtered_values: List[float] = [filtered_pose.q.y for filtered_pose in filtered_poses]
    plt.plot(filtered_values)
    plt.show()
    pass

