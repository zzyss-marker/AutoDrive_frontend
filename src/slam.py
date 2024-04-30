import logging
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam_data import Camera, Replica, RGBDImage


class PoseEstimator:
    """To estimate the camera pose based on sequential RGB images."""

    def __init__(self, K: np.ndarray):
        self.K = K
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.first_frame = True
        self.reference_kp = None
        self.reference_des = None
        self.reference_img = None

    def add_frame(self, img: np.ndarray) -> np.ndarray:
        """Add a new frame and compute the pose transformation matrix.
        :return transform_matrix: c2w
        """
        kp, des = self.orb.detectAndCompute(img, None)
        if self.first_frame:
            # 初始化参考帧的处理
            self.reference_kp = kp
            self.reference_des = des
            self.reference_img = img
            self.first_frame = False
            return np.eye(4)

        matches = self.matcher.match(self.reference_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        logging.info(f"Found {len(matches)} matches.")

        if len(matches) < 8:
            logging.warning("Not enough matches to find a reliable pose.")
            return None

        # 匹配点处理和Essential Matrix的计算...
        if E is None:
            logging.error("Failed to compute a valid Essential Matrix.")
            return None
        src_pts = np.float32(
            [self.reference_kp[m.queryIdx].pt for m in matches]
        ).reshape(-1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # 归一化
        def normalize_points(pts):
            mean = np.mean(pts, axis=0)
            std = np.std(pts)
            return (pts - mean) / std, mean, std

        src_pts_norm, src_mean, src_std = normalize_points(src_pts)
        dst_pts_norm, dst_mean, dst_std = normalize_points(dst_pts)

        E, mask = cv2.findEssentialMat(dst_pts, src_pts, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None:
            return None  # 检查Essential Matrix是否成功计算

        _, R, t, mask = cv2.recoverPose(E, dst_pts, src_pts, self.K)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = t.ravel()
        scale = np.eye(3) * src_std
        scale[2, 2] = 1
        T = np.eye(4)
        T[:3, :3] = scale
        T[:3, 3] = src_mean
        transform_matrix = np.linalg.inv(T) @ transform_matrix @ T
        return transform_matrix


class Mapper:
    def __init__(self, map_size=(500, 500), resolution=200, show_binary=True):
        self.map_size = map_size
        self.resolution = resolution
        self.origin_x = self.map_size[0] // 2
        self.origin_y = self.map_size[1] // 2

        self.heat_map = np.zeros(map_size)
        self.binary_map = np.zeros(map_size)
        self.show_binary = show_binary
        self.occupancy_threshold = 5  # 设置阈值，根据实际情况调整

    def build_map_2d(self, pcd: np.ndarray) -> None:
        """
        Build a 2D map from the given point cloud data.
        :param pcd: Nx3 numpy array of 3D points in world coordinates
        """
        self._build_heat_map(pcd)
        self._build_binary_map()

    def _build_heat_map(self, pcd: np.ndarray) -> None:

        # 转换坐标系到地图的索引
        x_indices = (pcd[:, 0] / self.resolution + self.origin_x).astype(int)
        y_indices = (pcd[:, 1] / self.resolution + self.origin_y).astype(int)

        # 过滤掉超出地图范围的点
        valid_indices = (
            (x_indices >= 0)
            & (x_indices < self.map_size[0])
            & (y_indices >= 0)
            & (y_indices < self.map_size[1])
        )
        x_indices = x_indices[valid_indices]
        y_indices = y_indices[valid_indices]

        # 用累加方式更新地图
        np.add.at(self.heat_map, (x_indices, y_indices), 1)

    def _build_binary_map(self) -> None:
        self.binary_map = np.where(self.heat_map > self.occupancy_threshold, 1, 0)

    def show(self) -> None:
        """展示地图，可选展示二值图或热力图。"""
        map_to_show = self.binary_map if self.show_binary else self.heat_map
        plt.imshow(
            map_to_show.T,
            origin="lower",
            cmap="gray" if self.show_binary else "hot",
            interpolation="nearest",
        )
        plt.colorbar()
        plt.draw()
        plt.pause(0.001)
        plt.clf()


class Slam2D:

    def __init__(self, input_folder, cfg_file: str, use_camera: bool = False) -> None:
        if use_camera:
            self.slam_data = Camera(input_folder, cfg_file)
        else:
            self.slam_data = Replica(input_folder, cfg_file)
        self.use_camera = use_camera
        self.tracker = PoseEstimator(self.slam_data.K)
        self.mapper = Mapper()
        # fps
        self.stamps = deque(maxlen=100)

    def run(self) -> None:
        """
        tracking and mapping
        """

        for i, rgb_d in enumerate(self.slam_data):
            rgb_d: RGBDImage
            start = cv2.getTickCount()
            pose = self.tracking(rgb_d.rgb) if self.use_camera else rgb_d.pose
            if pose is not None and not np.allclose(pose, 0):
                pcd_w = rgb_d.camera_to_world(pose, downsample_resolution=0.001)
                self.mapping(pcd_w)
            end = cv2.getTickCount()
            self.stamps.append((end - start) / cv2.getTickFrequency())
            if i % 50 == 0:
                logging.info(f"Average FPS: {1 / np.mean(self.stamps)}")
                self.mapper.show()
        self.mapper.show()

    def tracking(self, rgb_image: np.ndarray) -> np.ndarray | None:
        """
        tracking via rgp images
        :return c2w
        """
        pose = self.tracker.add_frame(rgb_image)
        if pose is None:
            logging.warning("Tracking failed or not enough matches.")
        return pose

    def mapping(self, pcd: np.ndarray) -> None:
        """
        update map via pdc
        :param pcd: point cloud in world coordinate
        """
        self.mapper.build_map_2d(pcd)
