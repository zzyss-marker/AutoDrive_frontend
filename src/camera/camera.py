import yaml
import numpy as np
import open3d as o3d
import cv2
from openni import openni2, utils

class CameraConfig:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def get_intrinsics(self):
        return self.config['camera_intrinsics']

class RGBDFrame:
    def __init__(self, rgb, depth):
        self.rgb = rgb
        self.depth = depth
    
    def depth_to_point_cloud(self, camera_intrinsics):
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        points = []
        height, width = self.depth.shape
        for v in range(height):
            for u in range(width):
                z = self.depth[v, u] / 1000.0  
                if z == 0: continue  # Skip no depth
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
        return np.asarray(points)

    def create_open3d_point_cloud(self, camera_intrinsics):
        points = self.depth_to_point_cloud(camera_intrinsics)
        if points.size == 0:
            print("No points were generated from depth data.")
            return None
        if np.isnan(points).any() or np.isinf(points).any():
            print("Points array contains NaN or Inf values.")
            return None
        print(f"Points array shape: {points.shape}, dtype: {points.dtype}")
        
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            return pcd
        except RuntimeError as e:
            print(f"Failed to create Open3D point cloud: {e}")
            return None

class Camera:
    def __init__(self, config_file):
        self.config = CameraConfig(config_file)
        openni2.initialize("/home/pixiu/Downloads/OrbbecViewer_1.1.13_202207221544_Linux")
        self.device = openni2.Device.open_any()
        self.depth_stream = self.device.create_depth_stream()
        self.depth_stream.start()
        self.capture = cv2.VideoCapture(2)
    
    def get_frame(self):
        if not self.capture.isOpened():
            print("Error opening video stream or file")
            return None
        ret, rgb_frame = self.capture.read()
        if not ret:
            print("Failed to grab a frame")
            return None
        depth_frame = self.get_depth_frame()
        return RGBDFrame(rgb_frame, depth_frame)
    
    def get_depth_frame(self):
        frame = self.depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        depth_array = np.ndarray((frame.height, frame.width), dtype=np.uint16, buffer=frame_data)
        return depth_array
    
    def cleanup(self):
        self.capture.release()
        self.depth_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()

def main():
    camera = Camera('camera_config.yaml')
    all_points = []  # List to accumulate all points from each frame
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
            pcd = frame.create_open3d_point_cloud(camera.config.get_intrinsics())
            if pcd is None:
                print("Failed to process the point cloud.")
                continue
            current_points = np.asarray(pcd.points)
            all_points.append(current_points)

            cv2.imshow('RGB Frame', frame.rgb)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    finally:
        camera.cleanup()
    
    if all_points:
        all_points = np.vstack(all_points)  
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        o3d.visualization.draw_geometries([combined_pcd])
    else:
        print("No points were collected.")

if __name__ == "__main__":
    main()
