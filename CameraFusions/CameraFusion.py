
from typing import Tuple, Annotated
import numpy as np
import numpy.typing as npt
import numpy.linalg as la


Position = Annotated[npt.NDArray[np.float64], (3,)]
Quaternion = Annotated[npt.NDArray[np.float64], (4,)]
Pose = Annotated[npt.NDArray[np.float64], (7,)]
Vector36D = Annotated[npt.NDArray[np.float64], (36,)]
Matrix3x3 = Annotated[npt.NDArray[np.float64], (3, 3)]
PoseCovariance = Annotated[npt.NDArray[np.float64], (6, 6)]




def position_fusion(pos1: Position, pos2: Position,
                    covariance1: Vector36D, covariance2: Vector36D) -> Tuple[Position, Matrix3x3]:
    cov1, cov2 = _create_pos_covariances(covariance1, covariance2)

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


def _create_pos_covariances(covariance1, covariance2) -> Tuple[Matrix3x3, Matrix3x3]:
    """Convert covariance matrices to position covariances only"""
    posecov1: PoseCovariance = covariance1.reshape(6, 6)
    posecov2: PoseCovariance = covariance2.reshape(6, 6)
    cov1: Matrix3x3 = posecov1[:3, :3]
    cov2: Matrix3x3 = posecov2[:3, :3]
    return cov1, cov2


def cholesky_inverse(mat: Matrix3x3) -> Matrix3x3:
    L = la.cholesky(mat)
    L_inv = la.solve(L, np.eye(3))
    return L_inv.T @ L_inv
