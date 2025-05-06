from src.Driver import Driver
from src.Graph import Graph
import numpy as np
import cv2 as cv


if __name__=="__main__":
    driver = Driver()
    trajectory, keypoints, frames = driver.run()
    graph = Graph()

    # Convert to NumPy for easier manipulation
    trajectory = np.array(trajectory)

    min_x, max_x = np.min(trajectory[:,0]), np.max(trajectory[:,0])
    all_kp = np.concatenate([frame[:, 1] for frame in keypoints])
    min_y, max_y = np.min(all_kp), np.max(all_kp)

    print()
    # for idx, frame_path in enumerate(frames):
    #     print(f"\rPlotting Map: {idx}/{len(frames)}", end=" ...", flush=True)
    #     img = cv.imread(frame_path)
        
    #     graph.plot_and_save_map(
    #         img=img,
    #         traj=trajectory[:idx+1],
    #         map_points=np.concatenate(keypoints[:idx+1], axis=0),
    #         X_bound=(min_x-30, max_x+30), y_bound=(min_y, max_y),
    #         idx=idx,
    #     ) 
    print()
    print("Plotting Trajectory")
    graph.plot_trajectory(trajectory)