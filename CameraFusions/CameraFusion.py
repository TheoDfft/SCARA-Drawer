
from typing import Tuple, Annotated
import numpy as np
import numpy.typing as npt
import numpy.linalg as la


Position = Annotated[npt.NDArray[np.float64], (3,)]
Quaternion = Annotated[npt.NDArray[np.float64], (4,)]
Pose = Annotated[npt.NDArray[np.float64], (7,)]
Vector36D = Annotated[npt.NDArray[np.float64], (36,)]
Matrix3x3 = Annotated[npt.NDArray[np.float64], (3, 3)]
Matrix4x4 = Annotated[npt.NDArray[np.float64], (4, 4)]
PoseCovariance = Annotated[npt.NDArray[np.float64], (6, 6)]


def pose_fusion(pose1: Pose, pose2: Pose, covariance1: Vector36D, covariance2: Vector36D) -> Tuple[Pose, Vector36D]:
    posecov1: PoseCovariance = covariance1.reshape(6, 6)
    posecov2: PoseCovariance = covariance2.reshape(6, 6)
    pos_cov1 = posecov1[:3, :3]
    pos_cov2 = posecov2[:3, :3]
    orientation_cov1 = posecov1[3:, 3:]
    orientation_cov2 = posecov1[3:, 3:]
    pos, pos_cov = position_fusion(pose1[:3], pose2[:3], pos_cov1, pos_cov2)

    quaternion1: Quaternion = pose1[3:]
    quaternion2: Quaternion = pose2[3:]
    orientation, orientation_cov = orientation_fusion(quaternion1, quaternion2,
                                                      orientation_cov1, orientation_cov2)

    zeros: Matrix3x3 = np.zeros((3, 3))
    covariance: Vector36D = np.block([[pos_cov, zeros],
                                      [zeros, orientation_cov]]).flatten()
    pose: Pose = np.concatenate(pos, orientation)
    return pose, covariance

def position_fusion(pos1: Position, pos2: Position,
                    cov1: Matrix3x3, cov2: Matrix3x3) -> Tuple[Position, Matrix3x3]:

    cov1_inv: Matrix3x3 = cholesky_inverse(cov1)
    cov2_inv: Matrix3x3 = cholesky_inverse(cov2)

    #Calculate weights of each camera using the covariances
    w1: float = 1/(np.trace(cov1) + 1e-9) #Avoiding division by 0
    w2: float = 1/(np.trace(cov2) + 1e-9) #Avoiding division by 0
    sumw: float = w1 + w2
    w1 = w1 / sumw
    w2 = w2 / sumw

    #Calculate the fused covariance and position
    cov: Matrix3x3 = cholesky_inverse(cov1_inv + cov2_inv)
    pos: Position = cov @ (((w1 * cov1_inv) @ pos1) + ((w2 * cov2_inv) @ pos2))

    return pos, cov


def orientation_fusion(q1: Quaternion, q2: Quaternion,
                       cov1: Matrix3x3, cov2: Matrix3x3) -> Tuple[Quaternion, Matrix3x3]:
    pass


def cholesky_inverse(mat: Matrix3x3) -> Matrix3x3:
    L = la.cholesky(mat)
    L_inv = la.solve(L, np.eye(3))
    return L_inv.T @ L_inv
