import logging

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
    def __init__(self, map_size=(500, 500), resolution=200, downsample_resolution=0.25):
        self.map_size = map_size
        self.resolution = resolution
        self.downsample_resolution = downsample_resolution
        self.map_2d = np.zeros(map_size)
        self.origin_x = self.map_size[0] // 2
        self.origin_y = self.map_size[1] // 2

    def build_map_2d(self, pcd: np.ndarray) -> None:
        # 在构建地图之前对点云进行下采样
        if self.downsample_resolution < 1.0:
            selected_indices = np.random.choice(
                len(pcd),
                size=int(len(pcd) * self.downsample_resolution),
                replace=False,
            )
            pcd = pcd[selected_indices]

        for point in pcd:
            x, y, z = point
            if z > 0:
                x_idx = int(x / self.resolution + self.origin_x)
                y_idx = int(y / self.resolution + self.origin_y)
                if 0 <= x_idx < self.map_size[0] and 0 <= y_idx < self.map_size[1]:
                    self.map_2d[x_idx, y_idx] += 1
                    logging.info(f"Added point to map at ({x_idx}, {y_idx}).")
                else:
                    logging.info(f"Point ({x_idx}, {y_idx}) out of map bounds.")


class Slam2D:

    def __init__(self, input_folder, cfg_file: str, use_camera: bool = False) -> None:
        if use_camera:
            self.slam_data = Camera(input_folder, cfg_file)
        else:
            self.slam_data = Replica(input_folder, cfg_file)
        self.use_camera = use_camera
        self.tracker = PoseEstimator(self.slam_data.K)
        self.mapper = Mapper()
        self.fig = None
        self.ax = None
        self.im = None

    def run(self) -> None:
        """
        tracking and mapping
        """
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.im = self.ax.imshow(
            np.zeros(self.mapper.map_size).T,
            origin="lower",
            cmap="gray",
            interpolation="nearest",
        )
        plt.show(block=False)
        for i, rgb_d in enumerate(self.slam_data):
            rgb_d: RGBDImage
            if self.use_camera:
                pose = self.tracking(rgb_d.rgb)
            else:
                pose = rgb_d.pose
            if pose is not None and not np.allclose(pose, 0):
                pcd_w = rgb_d.camera_to_world(pose)
                self.mapping(pcd_w)
            self.show()

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

    def show(self) -> None:
        """
        show the 2D map
        """
        plt.imshow(
            self.mapper.map_2d.T, origin="lower", cmap="hot", interpolation="nearest"
        )
        plt.colorbar()
        plt.draw()
        plt.pause(0.001)
        plt.clf()
