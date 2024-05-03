from collections import deque
from pathlib import Path
from timeit import default_timer

import numpy as np
from matplotlib import pyplot as plt

from component import PcdVisualizer
from component.tracker import Scan2ScanICP
from src.slam_data.dataset import Replica
from src.slam_data.Image import RGBDImage


class PointCloudProcessor(Scan2ScanICP):
    """
    Responsibilities:
    - Visualize the aggregated point cloud  vis real-time and with the real trajectory
    - Visualize the camera trajectory in 3D space real-time and compare it with real trajectory
    """

    def __init__(
        self,
        dataset_path,
        config_path,
        output_folder,
        max_images=None,
    ):
        super().__init__()
        self.data_loader = Replica(dataset_path, config_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.max_images = max_images or len(self.data_loader)
        # 2d map poses
        self.gt_poses: list[np.ndarray] = []
        self.estimated_poses: list[np.ndarray] = []
        self.stamps = deque(maxlen=20)
        self.vis = PcdVisualizer(intrinsic_matrix=self.data_loader.K)

    def process_pcd(self):

        for i, rgbd_image in enumerate(self.data_loader):
            start = default_timer()
            if i >= self.max_images:
                break
            # print(f"Processing image {i + 1}/{len(self.data_loader)}...")
            rgbd_image: RGBDImage
            # convert tensors to numpy arrays
            if rgbd_image.pose is None:
                raise ValueError("Pose is not available.")

            self.gt_poses.append(rgbd_image.pose)
            new_pcd = rgbd_image.depth_to_pointcloud(4)
            if not self.estimated_poses:
                estimate_pose = self.align_pcd(new_pcd, rgbd_image.pose)
            else:
                estimate_pose = self.align_pcd(new_pcd)

            self.stamps.append(1 / (default_timer() - start))
            fps = sum(self.stamps) / len(self.stamps)
            self.estimated_poses.append(estimate_pose)

            if i % 50 == 0:
                self.vis.update_render(new_pcd, estimate_pose)

            if i % 50 == 0:
                self.vis.vis_trajectory(
                    gt_poses=self.gt_poses,
                    estimated_poses=self.estimated_poses,
                    downsampling_resolution=self.downsampling_resolution,
                    fps=fps,
                )
        self.vis.close()

    #     if save:
    #         self.vis_final_and_save_pcd()
    #
    # def vis_final_and_save_pcd(self):
    #     if not self.accumulated_pcd:
    #         logging.warning("No point cloud to visualize.")
    #         return
    #     points_registered = np.concatenate(self.accumulated_pcd)
    #     pcd_o3d = o3d.geometry.PointCloud()
    #     pcd_o3d.points = o3d.utility.Vector3dVector(points_registered)
    #     print("Visualization of the complete point cloud...")
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window(window_name="Complete Point Cloud", width=1600, height=1200)
    #     vis.add_geometry(pcd_o3d)
    #     vis.run()
    #     vis.destroy_window()

    # # Save the point cloud to a .ply file
    # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # output_path = self.output_folder / f"aggregated_point_cloud_{current_time}.ply"
    # o3d.io.write_point_cloud(str(output_path), pcd_o3d, write_ascii=True)
    # print(f"Aggregated point cloud saved to {output_path}")

    # def visualize_ply(self, filepath):
    #     print(f"Loading and visualizing {filepath}...")
    #     pcd = o3d.io.read_point_cloud(filepath)
    #     o3d.visualization.draw_geometries([pcd])

    def visualize_trajectory(self):
        """vis 3d trajectory"""
        poses = [
            rgbd_image.pose.cpu().numpy()
            for rgbd_image in self.data_loader
            if rgbd_image.pose is not None
        ]
        translations = [pose[:3, 3] for pose in poses]

        # Plot the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        trajs = np.array(translations)
        ax.plot(trajs[:, 0], trajs[:, 1], trajs[:, 2], "-b")  # Blue line for trajectory

        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        plt.title("Camera Trajectory")
        plt.show()


if __name__ == "__main__":
    processor = PointCloudProcessor(
        dataset_path="/home/atticuszz/DevSpace/python/AutoDrive_frontend/Datasets/Replica/room0",
        config_path="/home/atticuszz/DevSpace/python/AutoDrive_frontend/Datasets/Replica/cam_params.json",
        output_folder="/home/atticuszz/DevSpace/python/AutoDrive_frontend/Datasets/Replica/pcl_output",
        max_images=2000,
    )
    pcd = processor.process_pcd()
    # processor.visualize_trajectory()
