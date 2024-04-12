import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np

from RGBD_image import Camera, RGBDImage


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
            self.reference_kp = kp
            self.reference_des = des
            self.reference_img = img
            self.first_frame = False
            return np.eye(4)  # 第一帧时返回单位矩阵
        matches = self.matcher.match(self.reference_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 8:
            return None  # 如果匹配点太少，则返回None

        src_pts = np.float32(
            [self.reference_kp[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        E, mask = cv2.findEssentialMat(dst_pts, src_pts, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None:
            return None  # 检查Essential Matrix是否成功计算

        _, R, t, mask = cv2.recoverPose(E, dst_pts, src_pts, self.K)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = t.ravel()
        return transform_matrix


class Mapper:
    def __init__(self, map_size=(200, 200), resolution=0.1):
        self.map_size = map_size
        self.resolution = resolution
        self.map_2d = np.zeros(map_size)
        self.origin_x = self.map_size[0] // 2
        self.origin_y = self.map_size[1] // 2

    def build_map_2d(self, pcd: np.ndarray) -> None:
        """
        :param pcd: (n,x,y,z) in wc
        """
        for point in pcd:
            x, y, z = point
            if z > 0:
                x_idx = int(x / self.resolution + self.origin_x)
                y_idx = int(y / self.resolution + self.origin_y)
                if 0 <= x_idx < self.map_size[0] and 0 <= y_idx < self.map_size[1]:
                    self.map_2d[x_idx, y_idx] += 1


class Slam2D:

    def __init__(self, cfg_file: str) -> None:
        self.camera = Camera(cfg_file)
        self.tracker = PoseEstimator(self.camera.K)
        self.mapper = Mapper()

    def run(self) -> None:
        """
        tracking and mapping
        """
        for rgb_d in self.camera:
            rgb_d: RGBDImage
            pose = self.tracking(rgb_d.rgb)
            if pose:
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
