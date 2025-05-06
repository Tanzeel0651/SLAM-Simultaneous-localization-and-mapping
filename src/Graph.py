import matplotlib.pyplot as plt
from headers.GraphModule import GraphModule
import cv2 as cv
import os
import numpy as np
import yaml

class Graph(GraphModule):
    def __init__(self):
        self.config = yaml.safe_load(open("config/configuration.yaml"))
        pass
    
    @staticmethod
    def normalize(arr):
        #arr = np.array(arr).squeeze()
        min_vals = np.min(arr, axis=0)
        max_vals = np.max(arr, axis=0)
        return (arr - min_vals) / (max_vals - min_vals + 1e-8)
    
    def draw_match(self, gray1, kp1, gray2, kp2, filtered_matches, new_img):
        match_img = cv.drawMatches(
                    img1 = gray1, 
                    keypoints1 = kp1, 
                    img2 = gray2, 
                    keypoints2 = kp2, 
                    matches1to2 = filtered_matches[:100], 
                    outImg = None,
                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(os.path.join(self.config["match_dir"], new_img+".png"), match_img)
        
    def plot_and_save_map(self, img, traj, map_points, X_bound, y_bound, idx):
        traj = np.array(traj)
        map_points = np.array(map_points)
        fig = plt.figure(figsize=(12, 5))
        ax_img = fig.add_subplot(1, 2, 1)
        ax_map = fig.add_subplot(1, 2, 2)
        
        # --- Image with keypoints ---
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ax_img.imshow(img_rgb)
        ax_img.set_title(f"Frame {idx}")
        ax_img.axis('off')
        
        # --- Map view ---
        if len(map_points) > 0:
            # For better visualization, only use a subset of map points (max 5000)
            if len(map_points) > 5000:
                # Random sample for visualization
                sample_idx = np.random.choice(len(map_points), 5000, replace=False)
                points_to_plot = map_points[sample_idx]
            else:
                points_to_plot = map_points
            
            # Plot map points with alpha transparency and smaller size
            # Using X and Z as the top-down view coordinates (X is forward in vehicle frame)
            ax_map.scatter(points_to_plot[:, 0], points_to_plot[:, 1], c='k', s=0.5, alpha=0.3, label='Keypoints')
        
        if len(traj) > 0:
            # Plot trajectory
            ax_map.plot(traj[:, 0], traj[:, 1], c='r', linewidth=2, label='Trajectory')
            ax_map.scatter(traj[-1, 0], traj[-1, 1], c='g', s=50, label='Current Position')
            
            # Add a heading arrow
            if len(traj) >= 2:
                dx = traj[-1, 0] - traj[-2, 0] 
                dy = traj[-1, 1] - traj[-2, 1]
                arrow_scale = 3.0  # Increased for better visibility
                ax_map.arrow(traj[-1, 0], traj[-1, 1], dx * arrow_scale, dy * arrow_scale,
                        head_width=0.5, head_length=0.7, fc='g', ec='g')
        
        ax_map.set_xlabel('X')
        ax_map.set_ylabel('Y')
        ax_map.set_title("SLAM Map (Top View)")
        #ax_map.legend(loc='upper right')
        
        # Set equal aspect ratio for meaningful spatial representation
        ax_map.set_aspect('equal')

        # Set limits for a better view
        # if len(traj) > 0:
        #     # Use fixed limits based on the typical scale of KITTI dataset
        # import pdb;pdb.set_trace()
        ax_map.set_xlim(X_bound)
        ax_map.set_ylim(y_bound)
    
        
        ax_map.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # Create the directory if it doesn't exist
        plt.savefig(os.path.join(self.config["map_dir"], f"{idx:03d}.png"))     
        plt.close()

    def plot_trajectory(self, traj):
        # config = yaml.safe_load(open("config/configuration.yaml"))
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        # Plot in vehicle coordinates: X=forward, Y=left, Z=up
        ax.plot(traj[:,0], traj[:,1], traj[:,2])
        ax.set_xlabel('X (forward)')
        ax.set_ylabel('Y (left)')
        ax.set_zlabel('Z (up)')
        
        # Set reasonable aspect ratio
        max_range = np.max([
            np.max(traj[:,0]) - np.min(traj[:,0]), 
            np.max(traj[:,1]) - np.min(traj[:,1]),
            np.max(traj[:,2]) - np.min(traj[:,2])
        ])
        mid_x = (np.max(traj[:,0]) + np.min(traj[:,0])) * 0.5
        mid_y = (np.max(traj[:,1]) + np.min(traj[:,1])) * 0.5
        mid_z = (np.max(traj[:,2]) + np.min(traj[:,2])) * 0.5
        ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
        ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
        ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
        
        # Add a view angle that shows forward motion clearly
        ax.view_init(elev=20, azim=-15, roll=0)
        
        plt.title('Visual Odometry Trajectory')
        plt.savefig(self.config["trajectory_path"])
        plt.close()