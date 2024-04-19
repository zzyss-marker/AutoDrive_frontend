import logging
from collections.abc import Generator

import cv2
import numpy as np

from utils import as_intrinsics_matrix, load_camera_cfg


class RGBDImage:
    def __init__(
        self, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, depth_scale: float
    ):
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            raise ValueError("rgb's h,w != depth's h,w")
        self.rgb = rgb
        self.depth = depth
        self.height = rgb.shape[0]
        self.width = rgb.shape[1]
        self.K = K
        self.scale = depth_scale

    def camera_to_world(self, c2w: np.ndarray) -> np.ndarray:
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
        return points_world

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


class Camera:
    def __init__(self, cfg_file: str):
        self.cfg = load_camera_cfg(cfg_file)["camera_params"]
        self.scale = self.cfg["png_depth_scale"]
        self.K = as_intrinsics_matrix(
            [self.cfg["fx"], self.cfg["fy"], self.cfg["cx"], self.cfg["cy"]]
        )

        openni2.initialize(
            "/home/pixiu/Downloads/OrbbecViewer_1.1.13_202207221544_Linux"
        )
        self.device = openni2.Device.open_any()
        self.depth_stream = self.device.create_depth_stream()
        self.depth_stream.start()
        self.capture = cv2.VideoCapture(2)

    def __iter__(self) -> Generator[RGBDImage, None, None]:
        """
        get RGBDImage via camera
        :return: RGBDImage
        """
        try:
            color = self._get_rgb()
            depth = self._get_depth()
            yield RGBDImage(color, depth, self.K, self.scale)
        except Exception as e:
            logging.exception(e)
        finally:
            self.shut_down()

    def _get_rgb(self) -> np.ndarray:
        """
        :return: rgb_frame.shape=(height,width,color)
        """
        ret, rgb_frame = self.capture.read()
        if not ret:
            raise ValueError("got None RGBDFrame")
        return rgb_frame

    def _get_depth(self) -> np.ndarray:
        """
        :return: depth_array.shape = (height,width)
        """
        frame = self.depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        depth_array = np.ndarray(
            (frame.height, frame.width), dtype=np.uint16, buffer=frame_data
        )
        return depth_array

    def shut_down(self):
        """
        release resource...
        """
        cv2.destroyAllWindows()
        self.capture.release()
        self.depth_stream.stop()
        openni2.unload()
