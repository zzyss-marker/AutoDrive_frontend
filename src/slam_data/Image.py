import numpy as np


class RGBDImage:
    def __init__(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        depth_scale: float,
        pose: np.ndarray | None = None,
    ):
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            raise ValueError("rgb's h,w != depth's h,w")
        self.rgb = rgb
        self.depth = depth
        self.height = rgb.shape[0]
        self.width = rgb.shape[1]
        self.K = K
        self.scale = depth_scale
        self.pose = pose

    def camera_to_world(
        self, c2w: np.ndarray, downsample_resolution: float = 0.25
    ) -> np.ndarray:
        """
        Transform points from camera coordinates to world coordinates using the c2w matrix.
        :param c2w: 4x4 transformation matrix from camera to world coordinates
        :return: Nx3 numpy array of transformed 3D points in world coordinates
        """
        # Add an additional row of ones to the points matrix for homogeneous coordinates
        points_camera = self._depth_to_pointcloud()
        points_homogeneous = np.hstack(
            (points_camera, np.ones((points_camera.shape[0], 1)))
        )

        # Transform the points using the c2w matrix
        points_world_homogeneous = points_homogeneous @ c2w.T

        # Remove the homogeneous coordinate before returning
        points_world = points_world_homogeneous[:, :3]

        # Downsample the points
        selected_indices = np.random.choice(
            len(points_world),
            size=int(len(points_world) * downsample_resolution),
            replace=False,
        )
        return points_world[selected_indices]

    def _depth_to_pointcloud(self) -> np.ndarray:
        """
        Convert the depth image to a 3D point cloud.
        Returns:
            points (Nx3 numpy array): The array of 3D points.
        """
        i_indices, j_indices = np.indices((self.height, self.width))
        depth_values = self.depth.astype(float) * self.scale

        # Transform to camera coordinates
        x = (j_indices - self.K[0, 2]) * depth_values / self.K[0, 0]
        y = (i_indices - self.K[1, 2]) * depth_values / self.K[1, 1]
        z = depth_values

        points = np.stack((x, y, z), axis=-1)
        return points.reshape(-1, 3)
