import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from open3d.visualization import Visualizer


class PcdVisualizer:

    def __init__(self, intrinsic_matrix: NDArray[np.int32], view_scale=1.0):
        self.o3d_vis = Visualizer()
        self.o3d_vis.create_window(
            window_name="Complete Point Cloud", width=1600, height=1200, visible=True
        )
        # Set camera parameters for following the camera pose
        self.view_control = self.o3d_vis.get_view_control()
        self.camera_params = o3d.camera.PinholeCameraParameters()
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=int(640 * view_scale),
            height=int(480 * view_scale),
            fx=intrinsic_matrix[0, 0] * view_scale,
            fy=intrinsic_matrix[1, 1] * view_scale,
            cx=intrinsic_matrix[0, 2] * view_scale,
            cy=intrinsic_matrix[1, 2] * view_scale,
        )
        self.camera_params.intrinsic = intrinsic

    def update_render(
        self,
        new_pcd: NDArray[np.float64],
        estimate_pose: NDArray[np.float64],
    ):
        new_pcd = o3d.utility.Vector3dVector(new_pcd)
        pcd_o3d = o3d.geometry.PointCloud(new_pcd)
        # pcd_o3d.paint_uniform_color([0.5, 0.5, 0.5])
        pcd_o3d.transform(estimate_pose)
        self.o3d_vis.add_geometry(pcd_o3d)
        self.o3d_vis.update_geometry(pcd_o3d)
        self._follow_camera(estimate_pose)
        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()

    def _follow_camera(self, c2w: NDArray[np.float64]):
        """Adjust the view control to follow a series of camera transformations."""

        self.camera_params.extrinsic = np.linalg.inv(
            c2w
        )  # Convert c2w to camera intrinsic
        self.view_control.convert_from_pinhole_camera_parameters(
            self.camera_params, allow_arbitrary=True
        )

    def close(self):
        self.o3d_vis.destroy_window()

    def vis_trajectory(
        self,
        gt_poses: list[NDArray],
        estimated_poses: list[NDArray],
        downsampling_resolution: float,
        fps: float,
    ) -> None:
        """Visualize the camera trajectory in 2D space."""
        gt_traj = np.array([pose[:3, 3] for pose in gt_poses])
        icp_traj = np.array([pose[:3, 3] for pose in estimated_poses])
        plt.clf()
        plt.title(f"Downsample ratio {downsampling_resolution}\nfps : {fps:.2f}")
        plt.plot(icp_traj[:, 0], icp_traj[:, 1], label="g-icp trajectory", linewidth=3)
        plt.legend()
        plt.plot(gt_traj[:, 0], gt_traj[:, 1], label="ground truth trajectory")
        plt.legend()
        plt.axis("equal")
        plt.pause(0.01)
