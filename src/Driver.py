
import os
import yaml
from .SLAM import Slam
from .Graph import Graph
import numpy as np
import cv2 as cv
from headers.DriverModule import DriverModule

class Driver(DriverModule):
    def __init__(self):
        self.slam = Slam()
        self.graph = Graph()

        self.config = yaml.safe_load(open("config/configuration.yaml"))
        self.paths = {
            "left_image_path": self.config["left_image_rgb_path"].replace("${dataset_path}", self.config["dataset_path"]),
            "right_image_path": self.config["right_image_rgb_path"].replace("${dataset_path}", self.config["dataset_path"]),
            "dataset_path": self.config["dataset_path"]
        }
        self.left_images = sorted(os.listdir(self.paths["left_image_path"]))
        self.right_images = sorted(os.listdir(self.paths["right_image_path"]))

        self.P0, self.P1, self.K = self.slam.intialize_camera_calibration()
        self.trajectory = [np.zeros(3)]
        self.all_keypoints = []

   
    def compute_matches(self, img_left_t, img_right_t, img_left_t1, idx):
        pts_left_t, pts_right_t, matches = self.slam.keypoint_extraction(img_left_t, img_right_t, self.left_images[idx].split(".")[0][-3:]+"-"+ self.right_images[idx].split(".")[0][-3:])
        # if pts_left_t is None or len(pts_left_t) < 20:
        #     print(f"Frame {idx}: Not enough stereo matches")
        #     self.trajectory.append(self.pose_T.copy())
        #     return 
        # import pdb;pdb.set_trace()
        pts_3d, pts_left_t = self.slam.triangulate_points(
                                        pts1 = pts_left_t, 
                                        pts2 = pts_right_t, 
                                        P0 = self.P0, P1 = self.P1)
        
        # if pts_3d is None or len(pts_3d) < 20:
        #     print(f"Frame {idx}: Not enough valid 3D points")
        #     self.trajectory.append(self.self.pose_T.copy())
        #     return
        
        # Step 2: Temporal matching
        prev_pts, next_pts = self.slam.match_keypoints_temporal(
                                        img1 = img_left_t, 
                                        img2 = img_left_t1, 
                                        prev_pts = pts_left_t)
        # if prev_pts is None or len(prev_pts) < 8:
        #     print(f"Frame {idx}: Not enough temporal matches")
        #     self.trajectory.append(self.self.pose_T.copy())
        #     if len(self.all_keypoints) > 0:
        #         self.all_keypoints.append(self.all_keypoints[-1])
        #     return
        
        # Find corresponding 3D points
        indices = []
        for pt in prev_pts:
            # Find the index of this point in pts_left_t
            dists = np.sqrt(np.sum((pts_left_t - pt)**2, axis=1))
            min_idx = np.argmin(dists)
            if dists[min_idx] < 2.0:  # 2-pixel threshold for considering it a match
                indices.append(min_idx)
        
        # if len(indices) < 8:
        #     print(f"Frame {idx}: Not enough matched 3D-2D pairs")
        #     self.trajectory.append(self.pose_T.copy())
        #     return 
        
        pts_3d_matched = pts_3d[indices]
        next_pts_matched = next_pts
        print(f"PnP input: {len(pts_3d_matched)} 3D-2D correspondences")
        return pts_3d_matched, next_pts_matched
    
    #@staticmethod
    def read_image(self, image_path, image):
        return cv.imread(os.path.join(image_path, image))

    def run(self, max_frames=float('inf')):
        frames = []
        idx = 0
        total = min(len(self.left_images), len(self.right_images), max_frames) - 1

        while idx < total:
            print(f"\nProcessing frame {idx}")

            img_left_t = self.read_image(self.paths["left_image_path"], self.left_images[idx])
            img_right_t = self.read_image(self.paths["right_image_path"], self.right_images[idx])
            img_left_t1 = self.read_image(self.paths["left_image_path"], self.left_images[idx+1])

            print(f"Image dimensions: {img_left_t.shape}")

            pts_3d_matched, next_pts_matched = self.compute_matches(img_left_t, img_right_t, img_left_t1, idx)
            vehicle_pose, map_keypoints = self.slam.pose_estimation(pts_3d_matched, next_pts_matched, self.K)

            self.trajectory.append(vehicle_pose.copy())
            self.all_keypoints.append(np.array(map_keypoints))
            frames.append(os.path.join(self.paths["left_image_path"], self.left_images[idx]))

            # Dynamic skip based on motion
            motion_threshold = 0.2  # meters
            if len(self.trajectory) >= 2:
                delta_pose = self.trajectory[-1] - self.trajectory[-2]
                distance = np.linalg.norm(delta_pose)
                if distance < motion_threshold:
                    print(f"Skipping next frame due to low motion: {distance:.3f}m")
                    idx += 2
                    continue

            idx += 1

        return self.trajectory, self.all_keypoints, frames
    

