
from abc import ABC, abstractmethod

class SLAMInterface(ABC):
    
    @abstractmethod
    def keypoint_extraction(self, img1, img2, new_img=None):
        pass

    @abstractmethod
    def triangulate_points(self, pts1, pts2, P0, P1, min_disp=1.0, max_disp=64.0):
        pass

    @abstractmethod
    def pose_estimation(self, pts_3d_matched, next_pts_matched, K):
        pass

    @abstractmethod
    def match_keypoints_temporal(img1, img2, prev_pts):
        pass

    @abstractmethod
    def intialize_camera_calibration(calib_file):
        pass
