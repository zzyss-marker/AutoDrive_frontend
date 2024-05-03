import logging
from collections import deque

import cv2
import numpy as np

from component import Mapper
from component.tracker import PoseEstimator
from slam_data import Camera, Replica, RGBDImage


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
                pcd_w = rgb_d.camera_to_world(pose, downsample_stride=4)
                self.mapping(pcd_w)
            end = cv2.getTickCount()
            self.stamps.append((end - start) / cv2.getTickFrequency())
            if i % 30 == 0:
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
