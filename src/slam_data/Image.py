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
            raise ValueError(
                "RGB's height and width must match Depth's height and width."
            )
        self.rgb = rgb
        self.depth = depth
        self.height = rgb.shape[0]
        self.width = rgb.shape[1]
        self.K = K
        self.scale = depth_scale
        self.pose = pose

    def camera_to_world(
        self, c2w: np.ndarray, downsample_resolution: float = 0.8
    ) -> np.ndarray:
        """
        Transform points from camera coordinates to world coordinates using the c2w matrix.
        :param c2w: 4x4 transformation matrix from camera to world coordinates
        :return: Nx3 numpy array of transformed 3D points in world coordinates
        """
        points_camera = self._depth_to_pointcloud(downsample_resolution)
        points_homogeneous = np.hstack(
            (points_camera, np.ones((points_camera.shape[0], 1)))
        )
        points_world_homogeneous = points_homogeneous @ c2w.T
        points_world = points_world_homogeneous[:, :3]
        return points_world

    def _depth_to_pointcloud(self, downsample_resolution: float = 0.8) -> np.ndarray:
        """
        Convert the depth image to a 3D point cloud based on a downsample resolution.
        :param downsample_resolution: Fraction of the total pixels to keep.
        :return: Nx3 numpy array of 3D points.
        """
        # Calculate downsample stride
        downsample_stride = max(1, int(np.sqrt(1 / downsample_resolution)))
        # print(downsample_stride)
        # Generate pixel indices
        i_indices, j_indices = np.indices(self.depth.shape)

        # Apply downsampling
        i_indices = i_indices[::downsample_stride, ::downsample_stride]
        j_indices = j_indices[::downsample_stride, ::downsample_stride]
        depth_downsampled = self.depth[::downsample_stride, ::downsample_stride]

        # Scale to meter
        depth_values = depth_downsampled.astype(float) / self.scale

        # Transform to camera coordinates
        x = (j_indices - self.K[0, 2]) * depth_values / self.K[0, 0]
        y = (i_indices - self.K[1, 2]) * depth_values / self.K[1, 1]
        z = depth_values

        points = np.stack((x, y, z), axis=-1)
        return points.reshape(-1, 3)
