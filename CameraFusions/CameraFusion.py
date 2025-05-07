from typing import Tuple, Annotated, Union
import numpy as np
import numpy.typing as npt
import numpy.linalg as la


Matrix3x3 = Annotated[npt.NDArray[np.float64], (3, 3)]
Matrix4x4 = Annotated[npt.NDArray[np.float64], (4, 4)]
PoseCovariance = Annotated[npt.NDArray[np.float64], (6, 6)]


class Position:
    def __init__(self, x: float, y: float, z: float):
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_array(array) -> 'Position':
        return Position(array[0], array[1], array[2])

    def __matmul__(self, other: Union['Position', np.ndarray]) -> float:
        if isinstance(other, Position):
            return self.x * other.x + self.y * other.y + self.z * other.z
        elif isinstance(other, np.ndarray):
            return np.dot(self.to_array(), other)
        else:
            raise TypeError(f"Unsupported operand type for @: 'Position' and '{type(other).__name__}'")

class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w: float = w
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def to_array(self):
        return np.array([self.w, self.x, self.y, self.z])

    @staticmethod
    def from_array(array) -> 'Quaternion':
        return Quaternion(array[0], array[1], array[2], array[3])

    def __matmul__(self, other: Union['Quaternion', np.ndarray]) -> float:
        if isinstance(other, Quaternion):
            return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
        elif isinstance(other, np.ndarray):
            return np.dot(self.to_array(), other)
        else:
            raise TypeError(f"Unsupported operand type for @: 'Quaternion' and '{type(other).__name__}'")

    def __truediv__(self, scalar: float) -> 'Quaternion':
        return Quaternion(self.w / scalar, self.x / scalar,
                           self.y / scalar, self.z / scalar)

    def as_rotation_vector(self) -> np.ndarray:
        norm = np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        w = self.w / norm
        x = self.x / norm
        y = self.y / norm
        z = self.z / norm

        # Compute rotation angle
        theta = 2 * np.arccos(np.clip(w, -1.0, 1.0))

        # Handle small angles to avoid division by zero
        epsilon = 1e-7
        sin_half_theta = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        if sin_half_theta < epsilon:
            return np.zeros(3)

        # Compute rotation vector components
        scale = theta / sin_half_theta
        return np.array([x * scale, y * scale, z * scale])

class Pose:
    def __init__(self, position: Position, q: Quaternion):
        self.Position: Position = position
        self.q: Quaternion = q


class OnlinePoseCovariance:
    def __init__(self):
        self.forgetting_factor: float = 0.95
        self.n: int = 0
        self.mean: float = np.zeros(6)
        self.M2: PoseCovariance = np.zeros((6, 6))  # Sum of squared deviations

    def update(self, pose: Pose):
        rot_vector: np.ndarray = pose.q.as_rotation_vector()
        x = np.concatenate([pose.Position.to_array(), rot_vector])

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)

        #Forgetting factor
        self.mean = self.forgetting_factor * self.mean + (1 - self.forgetting_factor) * x
        self.M2 = self.forgetting_factor * self.M2 + (1 - self.forgetting_factor) * np.outer(x - self.mean, x - self.mean)

    @property
    def covariance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else np.zeros((6, 6))


def pose_fusion(pose1: Pose, pose2: Pose, covariance1: PoseCovariance, covariance2: PoseCovariance) -> Tuple[Pose, PoseCovariance]:
    pos_cov1 = covariance1[:3, :3]
    pos_cov2 = covariance2[:3, :3]
    orientation_cov1 = covariance1[3:, 3:]
    orientation_cov2 = covariance2[3:, 3:]
    pos, pos_cov = position_fusion(pose1.Position, pose2.Position, pos_cov1, pos_cov2)

    quaternion1: Quaternion = pose1.q
    quaternion2: Quaternion = pose2.q
    orientation, orientation_cov = orientation_fusion(quaternion1, quaternion2,
                                                      orientation_cov1, orientation_cov2)

    zeros: Matrix3x3 = np.zeros((3, 3))
    covariance: PoseCovariance = np.block([[pos_cov, zeros],
                                           [zeros, orientation_cov]])
    pose: Pose = Pose(pos, orientation)
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
    pos: Position = Position.from_array(cov @ (((w1 * cov1_inv) @ pos1) + ((w2 * cov2_inv) @ pos2)))

    return pos, cov


def orientation_fusion(q1: Quaternion, q2: Quaternion,
                       cov1: Matrix3x3, cov2: Matrix3x3) -> Tuple[Quaternion, Matrix3x3]:
    #Normalize the quaternions
    q1 = q1 / np.linalg.norm(q1.to_array())
    q2 = q2 / np.linalg.norm(q2.to_array())

    # Antipodal check using dot product threshold
    if q1 @ q2 < 0:
        q2 = -q2  # Flip to same hemisphere

    #Regularizing covariances to ensure invertibility
    cov1 += 1e-6 * np.eye(3)
    cov2 += 1e-6 * np.eye(3)

    #Using trace
    w1: float = 1/(np.trace(cov1) + 1e-9)
    w2: float = 1/(np.trace(cov2) + 1e-9)

    ##################TODO CONSIDER AFTER A LOT OF DATA FOR BUILDING THE COVARIANCE MATRICES
    # Using det
    # w1: float = 1 / (np.linalg.det(cov1) + 1e-9)
    # w2: float = 1 / (np.linalg.det(cov2) + 1e-9)

    #Normalize the weights
    total: float = w1 + w2
    w1 /= total
    w2 /= total

    M: Matrix4x4 = w1 * np.outer(q1, q1) + w2 * np.outer(q2, q2) # M = V*Delta*V_transpose

    # Eigen decomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
        # Fallback to simple averaging if decomposition fails
        return (q1 + q2) / np.linalg.norm(q1 + q2)
    # Taking the last EigenVector = the Eigenvector respective to the largest Eigenvalue
    q: Quaternion = Quaternion.from_array(eigenvectors[:, np.argmax(eigenvalues)])
    # Normalize the quaternion
    q /= np.linalg.norm(q)

    # Ensure consistent hemisphere with first quaternion
    if np.dot(q, q1) < 0:
        q *= -1

    cov: Matrix3x3 = _fuse_orientation_cov(cov1, cov2)

    return q, cov


def _fuse_orientation_cov(cov1: Matrix3x3, cov2: Matrix3x3) -> Matrix3x3:
    # Calculate weights
    w1: float = 1 / (np.linalg.det(cov1) + 1e-9)
    w2: float = 1 / (np.linalg.det(cov2) + 1e-9)
    total: float = w1 + w2
    return (w1 * cov1 + w2 * cov2) / total


def cholesky_inverse(mat: Matrix3x3) -> Matrix3x3:
    """Inverse matrix calculation using the Cholesky factorization"""
    L = la.cholesky(mat)
    L_inv = la.solve(L, np.eye(3))
    return L_inv.T @ L_inv

