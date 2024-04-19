from datetime import datetime
from pathlib import Path
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

from slam_data import Replica, RGBDImage

class PointCloudProcessor:
    def __init__(self, dataset_path, config_path, output_folder, max_images=None, voxel_size=0.015):
        self.data_loader = Replica(dataset_path, config_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.max_images = max_images
        self.voxel_size = voxel_size

    def process_pcd(self, vis=True):
        aggregated_pcd = o3d.geometry.PointCloud()
        plt.figure()
        plt.ion()  # 开启交互模式

        for i, rgbd_image in enumerate(self.data_loader):
            if self.max_images and i >= self.max_images:
                break
            print(f"Processing image {i + 1}/{len(self.data_loader)}...")

            # 创建 Open3D RGBDImage
            color_o3d = o3d.geometry.Image(rgbd_image.rgb.astype(np.uint8))
            depth_o3d = o3d.geometry.Image(rgbd_image.depth.astype(np.uint16))
            rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=1000.0,
                convert_rgb_to_intensity=False
            )

            # 从 RGBD 图像创建点云
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_o3d,
                intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    rgbd_image.width,
                    rgbd_image.height,
                    rgbd_image.K[0, 0],
                    rgbd_image.K[1, 1],
                    rgbd_image.K[0, 2],
                    rgbd_image.K[1, 2]
                )
            )

            if rgbd_image.pose is not None:
                pcd.transform(rgbd_image.pose)

            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            aggregated_pcd += pcd

            # 更新二维占用网格图
            if i%20 ==0:
                self.update_occupancy_grid(aggregated_pcd, (-0.2, 0.2), 0.05)

        plt.ioff()  # 关闭交互模式
        if vis:
            self.visualize_and_save_pcd(aggregated_pcd)

    def update_occupancy_grid(self, pcd, height_range, grid_size):
        points = np.asarray(pcd.points)
        mask = (points[:, 2] > height_range[0]) & (points[:, 2] < height_range[1])
        points_filtered = points[mask][:, :2]

        # 确定网格维度
        min_x, min_y = points_filtered.min(axis=0)
        max_x, max_y = points_filtered.max(axis=0)
        grid_x = int((max_x - min_x) / grid_size) + 1
        grid_y = int((max_y - min_y) / grid_size) + 1
        grid = np.zeros((grid_y, grid_x), dtype=np.uint8)

        for x, y in (points_filtered - [min_x, min_y]) / grid_size:
            grid[int(y), int(x)] = 1  # 标记为占用

        plt.clf()  # 清除当前图形
        plt.imshow(grid, cmap='gray')
        plt.gca().invert_yaxis()
        plt.draw()
        plt.pause(0.001)  # 暂停一下以更新绘图

    def visualize_and_save_pcd(self, pcd):
        # Visualization
        print("Visualization of the complete point cloud...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Complete Point Cloud", width=1600, height=1200)
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

        # Save the point cloud to a .ply file
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = self.output_folder / f"aggregated_point_cloud_{current_time}.ply"
        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=True)
        print(f"Aggregated point cloud saved to {output_path}")

    def visualize_ply(self, filepath):
        print(f"Loading and visualizing {filepath}...")
        pcd = o3d.io.read_point_cloud(filepath)
        o3d.visualization.draw_geometries([pcd])

    def create_occupancy_grid(self, pcd, height_range, grid_size):
        print("creating occupancy_grid...")
        points = np.asarray(pcd.points)
        mask = (points[:, 2] > height_range[0]) & (points[:, 2] < height_range[1])
        points_filtered = points[mask][:, :2]  # Take only X and Y

        # Create the grid
        min_x, min_y = points_filtered.min(axis=0)
        max_x, max_y = points_filtered.max(axis=0)
        grid_x = int((max_x - min_x) / grid_size) + 1
        grid_y = int((max_y - min_y) / grid_size) + 1
        grid = np.zeros((grid_y, grid_x), dtype=np.uint8)

        for x, y in (points_filtered - [min_x, min_y]) / grid_size:
            grid[int(y), int(x)] = 1  # Mark the cell as occupied

        # Save the grid as an image
        plt.imshow(grid, cmap='gray')
        plt.gca().invert_yaxis()
        plt.savefig(self.output_folder / f"occupancy_grid_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        print("Occupancy grid saved.")

if __name__ == "__main__":
    processor = PointCloudProcessor(
        dataset_path='/home/atticuszz/DevSpace/python/AutoDrive_frontend/Datasets/Replica/room0',
        config_path='/home/atticuszz/DevSpace/python/AutoDrive_frontend/Datasets/Replica/cam_params.json',
        output_folder='/home/atticuszz/DevSpace/python/AutoDrive_frontend/Datasets/Replica/Output',
        max_images=500,
        voxel_size=0.015
    )
    pcd = processor.process_pcd(False)
    # # 创建和保存二维占用网格图
    # # 这通常在 process_pcd 方法的末尾自动调用，这里是显示如何单独调用
    # # 假设已有聚合点云 saved_pcd 从之前的处理中
    # # 这里直接使用processor里处理好的点云，具体处理请根据实际情况调整
    # # 选择高度范围和网格大小
    # height_range = (-0.2, 0.2)  # 仅考虑Z轴在-0.2到0.2范围内的点
    # grid_size = 0.05  # 网格大小设置为5厘米
    #
    # # 假设已有 aggregated_pcd 可用，可以通过成员访问或者其他方式获取
    # # 以下只是调用示例，实际使用时需要确保 aggregated_pcd 是可用的
    # processor.create_occupancy_grid(pcd, height_range, grid_size)

