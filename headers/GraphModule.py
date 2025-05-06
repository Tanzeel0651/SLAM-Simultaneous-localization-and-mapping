
from abc import ABC, abstractmethod

class GraphModule(ABC):
    
    @abstractmethod
    def plot_and_save_map(img, traj, map_points, idx):
        pass

    @abstractmethod
    def plot_trajectory():
        pass

    @abstractmethod
    def draw_match():
        pass