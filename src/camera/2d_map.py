import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml
from openni import openni2


# 定义相机配置类，用于加载相机参数
class CameraConfig:
    def __init__(self, config_file: str):
        with open(config_file) as file:
            self.config = yaml.safe_load(file)

    # 获取相机内参
    def get_intrinsics(self) -> dict:
        return self.config["camera_intrinsics"]


# 定义RGBD帧类，包含RGB图和深度图
class RGBDFrame:
    def __init__(self, rgb: np.ndarray, depth: np.ndarray):
        self.rgb = rgb
        self.depth = depth

    # 深度图转点云
    def depth_to_point_cloud(self, camera_intrinsics: dict) -> np.ndarray:
        fx = camera_intrinsics["fx"]
        fy = camera_intrinsics["fy"]
        cx = camera_intrinsics["cx"]
        cy = camera_intrinsics["cy"]
        points = []
        height, width = self.depth.shape
        for v in range(height):
            for u in range(width):
                z = self.depth[v, u] / 1000.0  # 将深度从毫米转换为米
                if z == 0:
                    continue  # 跳过没有深度的点
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
        return np.asarray(points)

    # 创建Open3D点云
    def create_open3d_point_cloud(
        self, camera_intrinsics: dict
    ) -> o3d.geometry.PointCloud:
        points = self.depth_to_point_cloud(camera_intrinsics)
        if points.size == 0:
            print("深度数据没有生成任何点。")
            return None
        print(f"点数组形状: {points.shape}, 数据类型: {points.dtype}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd


# 定义相机类，负责管理相机配置和数据流


class Camera:
    def __init__(self, config_file: str):
        self.config = CameraConfig(config_file)
        openni2.initialize(
            "/home/pixiu/Downloads/OrbbecViewer_1.1.13_202207221544_Linux"
        )
        self.device = openni2.Device.open_any()
        self.depth_stream = self.device.create_depth_stream()
        self.depth_stream.start()
        self.capture = cv2.VideoCapture(2)

    # 获取一帧数据
    def get_frame(self) -> RGBDFrame:
        ret, rgb_frame = self.capture.read()
        if not ret:
            raise ValueError("got None RGBDFrame")
        depth_frame = self.get_depth_frame()
        return RGBDFrame(rgb_frame, depth_frame)

    # 获取深度帧
    def get_depth_frame(self) -> np.ndarray:
        frame = self.depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        depth_array = np.ndarray(
            (frame.height, frame.width), dtype=np.uint16, buffer=frame_data
        )
        return depth_array

    # 清理，关闭数据流和窗口
    def cleanup(self):
        self.capture.release()
        self.depth_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()


# solvePnp return martrix to move and spin
# class CameraPose3D:
#     def __init__(self):
#         self.R = np.eye(3)  # 旋转矩阵初始化为单位矩阵
#         self.t = np.zeros((3, 1))  # 平移向量初始化为零向量

#     def update_pose(self, delta_R, delta_t):
#         # 更新相机位姿，delta_R为旋转矩阵，delta_t为平移向量
#         self.R = np.dot(delta_R, self.R)
#         self.t += delta_t


#     def transform_point(self, point):
#         # 根据当前相机位姿转换点的坐标
#         point_transformed = np.dot(self.R, point.reshape(3, 1)) + self.t
#         return point_transformed.flatten()
class CameraPose3D:
    def __init__(self):
        self.R = np.eye(3)  # 旋转矩阵初始化为单位矩阵
        self.t = np.zeros((3, 1))  # 平移向量初始化为零向量

    def update_pose(self, rvec, tvec):
        # 将旋转向量转换为旋转矩阵
        delta_R, _ = cv2.Rodrigues(rvec)
        # 旋转矩阵和平移向量更新
        self.R = delta_R @ self.R
        self.t = self.t + tvec

    def transform_point(self, point):
        # 根据当前相机位姿转换点的坐标
        return (self.R @ point.reshape(3, 1) + self.t).flatten()


# 点云转2D地图
def point_cloud_to_2d_map(
    points: np.ndarray, resolution: float = 0.05, width: int = 200, height: int = 200
) -> np.ndarray:
    occupancy_map = np.zeros((height, width))
    origin_x, origin_y = width // 2, height // 2
    for point in points:
        x, z = point[0], point[2]
        grid_x = int(x / resolution) + origin_x
        grid_z = int(z / resolution) + origin_y
        if 0 <= grid_x < width and 0 <= grid_z < height:
            occupancy_map[grid_z, grid_x] = 1
    return occupancy_map


# 将转换后的3D点投影到二终地图上
def update_global_map(
    global_map, transformed_points, resolution=0.05, width=200, height=200
):
    origin_x, origin_y = width // 2, height // 2
    for point in transformed_points:
        x, z = point[0], point[2]
        # 计算点在地图上的坐标
        grid_x = int(x / resolution) + origin_x
        grid_z = int(z / resolution) + origin_y
        # 确保坐标在地图范围内
        if 0 <= grid_x < width and 0 <= grid_z < height:
            global_map[grid_z, grid_x] = 1  # 标记为占用

    return global_map


def main():
    camera = Camera("camera_config.yaml")
    camera_pose = CameraPose3D()
    global_map = np.zeros((200, 200))

    # 加载相机内参
    camera_intrinsics = camera.config.get_intrinsics()
    camera_matrix = np.array(
        [
            [camera_intrinsics["fx"], 0, camera_intrinsics["cx"]],
            [0, camera_intrinsics["fy"], camera_intrinsics["cy"]],
            [0, 0, 1],
        ]
    )
    dist_coeffs = np.zeros((4, 1))  # 假设没有畸变

    # 初始化ORB和BFMatcher
    orb = cv2.ORB_create(nfeatures=100000000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_frame = None
    prev_des = None
    prev_kp = None

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue

            kp, des = orb.detectAndCompute(frame.rgb, None)

            # 确保描述符不为空并且上一帧的描述符也不为空
            if (
                des is not None
                and prev_des is not None
                and len(des) > 0
                and len(prev_des) > 0
            ):
                des = des.astype(np.float32)
                prev_des = prev_des.astype(np.float32)

                # 特征点匹配
                matches = bf.match(des, prev_des)
                matches = sorted(matches, key=lambda x: x.distance)

                if matches:
                    good_matches = matches[:50]  # 获取前50个最好的匹配项

                    obj_points = []  # 3D points in prev_frame
                    img_points = []  # 2D points in current frame

                    for match in good_matches:
                        # 获取匹配点对应的索引
                        img_idx = match.queryIdx
                        obj_idx = match.trainIdx

                        # 获取2D点的坐标
                        (x, y) = kp[img_idx].pt

                        # 使用深度信息转换为3D点
                        z = prev_frame.depth[
                            int(prev_kp[obj_idx].pt[1]), int(prev_kp[obj_idx].pt[0])
                        ]
                        if z == 0:  # 深度为0表示无效点
                            continue

                        z /= 1000.0  # 深度单位从毫米转为米
                        x = (x - camera_intrinsics["cx"]) * z / camera_intrinsics["fx"]
                        y = (y - camera_intrinsics["cy"]) * z / camera_intrinsics["fy"]

                        obj_points.append([x, y, z])
                        img_points.append(kp[img_idx].pt)

                    obj_points = np.asarray(obj_points, dtype=np.float32)
                    img_points = np.asarray(img_points, dtype=np.float32)

                    if len(obj_points) >= 4 and len(img_points) >= 4:
                        # 使用solvePnP进行姿态估计
                        _, rvec, tvec = cv2.solvePnP(
                            obj_points, img_points, camera_matrix, dist_coeffs
                        )
                        camera_pose.update_pose(rvec, tvec)

                        # 将3D点根据当前相机位姿转换到全局坐标系中
                        transformed_points = [
                            camera_pose.transform_point(point) for point in obj_points
                        ]

                        # 更新二维地图
                        global_map = update_global_map(global_map, transformed_points)

                        plt.imshow(global_map, cmap="gray")
                        plt.pause(0.05)
                else:
                    print("没有找到足够的匹配项。")
                    continue
            else:
                print("描述符为空或不足以进行匹配。")
                continue

            prev_frame = frame
            prev_des = des
            prev_kp = kp

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        camera.cleanup()


if __name__ == "__main__":
    main()
