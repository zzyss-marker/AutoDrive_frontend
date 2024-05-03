import numpy as np
from matplotlib import pyplot as plt


class Mapper:
    def __init__(self, map_size=(300, 250), resolution=0.05, show_binary=True):
        self.map_size = map_size
        self.resolution = resolution
        self.origin_x = self.map_size[0] // 2
        self.origin_y = self.map_size[1] // 2

        self.heat_map = np.zeros(map_size)
        self.binary_map = np.zeros(map_size)
        self.show_binary = show_binary
        self.occupancy_threshold = 8000  # 设置阈值，根据实际情况调整

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
        self.binary_map = np.where(self.heat_map < self.occupancy_threshold, 1, 0)

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
