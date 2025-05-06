
import os
import yaml
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .Graph import Graph

from headers.SLAMModule import SLAMInterface

class Slam(SLAMInterface):
    def __init__(self):
        self.graph = Graph()
        self.pose_R = np.eye(3)
        self.pose_T = np.zeros(3)
        self.config = yaml.safe_load(open("config/configuration.yaml"))

    def keypoint_extraction(self, img1, img2, new_img=None):
        #super().extract_keypoints(img1, img2, new_img)
        # Convert images to grayscale if needed
        if len(img1.shape) == 3:
            gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            gray2 = img2
        
        # Create ORB detector with more features and adjusted settings
        orb = cv.ORB_create(nfeatures=3000, scaleFactor=1.2, edgeThreshold=31)
        
        # Detect keypoints and compute descriptors
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        print(f"Keypoints found: {len(kp1)} in left, {len(kp2)} in right image")
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, None, None
        
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        
        # Sort by distance and filter
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:min(1500, len(matches))]
        
        # Extract points from matches
        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
        
        # Additional epipolar constraint for rectified stereo pairs
        # Points should have similar y-coordinates
        y_diff = np.abs(pts1[:,1] - pts2[:,1])
        epipolar_mask = y_diff < 5.0  # 5 pixel tolerance for y-difference
        
        if np.sum(epipolar_mask) < 20:
            print("Not enough matches satisfying epipolar constraint")
            return None, None, None
        
        pts1 = pts1[epipolar_mask]
        pts2 = pts2[epipolar_mask]
        filtered_matches = [m for i, m in enumerate(good_matches) if epipolar_mask[i]]
        
        print(f"Filtered matches: {len(filtered_matches)} after epipolar constraint")
        
        # Optional: Draw and save the matches for debugging
        if len(filtered_matches) > 0 and new_img:
            self.graph.draw_match(
                gray1, kp1, gray2, kp2, filtered_matches[:100], new_img
            )

        
        return (pts1, pts2, filtered_matches)
    
    def triangulate_points(self, pts1, pts2, P0, P1, min_disp=1.0, max_disp=64.0):
        """Triangulate 3D points from stereo image pairs"""
        # Calculate disparity (x1 - x2) for each point
        disparity = pts1[:,0] - pts2[:,0]
        
        # Filter based on disparity range
        valid_disp = (disparity > min_disp) & (disparity < max_disp)
        if np.sum(valid_disp) < 20:
            print(f"Not enough points with valid disparity (min: {min_disp}, max: {max_disp})")
            return None, None
        
        pts1_valid = pts1[valid_disp]
        pts2_valid = pts2[valid_disp]
        
        # Reshape points for cv.triangulatePoints
        pts1_valid = pts1_valid.reshape(-1,1,2).astype(np.float32)
        pts2_valid = pts2_valid.reshape(-1,1,2).astype(np.float32)
        
        # Triangulate points (4xN homogenous coordinates)
        pts4D = cv.triangulatePoints(P0, P1, pts1_valid, pts2_valid)
        
        # Convert to 3D (Nx3)
        pts3D = pts4D[:3]/pts4D[3]
        pts3D = pts3D.T
        
        # Filter points with reasonable Z values (0.5m to 100m)
        valid = (pts3D[:,2] > 0.5) & (pts3D[:,2] < 100)
        print(f"Valid 3D points after depth filtering: {np.sum(valid)}/{len(pts3D)}")
        
        if np.sum(valid) < 20:
            print("Not enough valid 3D points after depth filtering")
            return None, None
        
        pts3D = pts3D[valid]
        pts1_out = pts1_valid.reshape(-1,2)[valid]
        
        # Print statistics for debugging
        depths = pts3D[:,2]
        print(f"Depth stats - Min: {np.min(depths):.2f}m, Max: {np.max(depths):.2f}m, Mean: {np.mean(depths):.2f}m")
        
        return pts3D, pts1_out


    def match_keypoints_temporal(self, img1, img2, prev_pts):
        # Convert images to grayscale if needed
        if len(img1.shape) == 3:
            prev_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            next_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        else:
            prev_gray = img1
            next_gray = img2
        
        # Track features using optical flow
        next_pts, status, _ = cv.calcOpticalFlowPyrLK(
            prev_gray, next_gray,
            prev_pts.reshape(-1,1,2).astype(np.float32),
            None,
            winSize=(21,21),
            maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # Filter valid points
        valid = status.ravel() == 1
        valid_count = np.sum(valid)
        print(f"Temporal tracking: {valid_count}/{len(prev_pts)} points tracked successfully")
        
        if valid_count < 10:
            return None, None
        
        return prev_pts[valid], next_pts.reshape(-1,2)[valid]
    

    def read_images(left_images_list, right_images_list, path_dir, idx):
        left_dir = "image02"
        right_dir = "image03"
        return (cv.imread(os.path.join(path_dir, left_dir, left_images_list[idx])),
                cv.imread(os.path.join(path_dir, right_dir, right_images_list[idx])),
                cv.imread(os.path.join(path_dir, left_dir, left_images_list[idx+1])))
    

    def intialize_camera_calibration(self):
        if os.path.exists(self.config["calib"]):
            calib_file = yaml.safe_load(open(self.config["calib"].replace("${dataset_path}", self.config["dataset_path"]), 'r'))
            
            P0 = np.array([float(x) for x in calib_file["P_rect_02"].split()], dtype=np.float32).reshape(3,4)            
            
            P1 = np.array([float(x) for x in calib_file["P_rect_03"].split()], dtype=np.float32).reshape(3,4)
        
        else:
            P0 = np.array([7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00,
                0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00,
                0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]).reshape(3,4)

            P1 = np.array([7.215377e+02, 0.000000e+00, 6.095593e+02, -3.875744e+02,
                0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00,
                0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]).reshape(3,4)
            
        K = P0[:,:3]
        return P0, P1, K
    

    def pose_estimation(self, pts_3d_matched, next_pts_matched, K):
        success, rvec, tvec, inliers = cv.solvePnPRansac(
            pts_3d_matched.reshape(-1,1,3),
            next_pts_matched.reshape(-1,1,2),
            K, None,
            iterationsCount=500,
            reprojectionError=2.0,
            confidence=0.99,
            flags=cv.SOLVEPNP_ITERATIVE
        )

        print(f"PnP successful with {len(inliers)} inliers")

        # if not success or inliers is None or len(inliers) < 8:
        #     print(f"Frame {i}: PnP failed")
        #     self.traj.append(pose_T.copy())
        #     continue
            
        # Step 4: Update camera pose
        R_est, _ = cv.Rodrigues(rvec)
        self.pose_T = self.pose_T + self.pose_R @ tvec.ravel()
        self.pose_R = self.pose_R @ R_est
        
        # Step 5: Store trajectory
        # Convert camera coordinates to vehicle coordinates
        # KITTI: X=forward, Y=left, Z=up
        # Camera: Z=forward, X=right, Y=down
        
        # FIX: Corrected coordinate transformation
        vehicle_pose = np.array([-self.pose_T[2], -self.pose_T[0], -self.pose_T[1]])  # [-Z, -X, -Y] for forward motion
        
        pts_3d_world = (self.pose_R @ pts_3d_matched.T).T + self.pose_T.reshape(1, 3)
        map_keypoints = np.array([-pts_3d_world[:,2], -pts_3d_world[:,0]]).T
        
        return vehicle_pose, map_keypoints
    
    

        

