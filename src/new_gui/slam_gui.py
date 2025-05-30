import matplotlib.pyplot as plt
import numpy as np
import dearpygui.dearpygui as dpg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from evo.tools import plot
from evo.core.trajectory import PoseTrajectory3D
import time
import cv2
from multiprocessing import Queue
from src.new_gui.gui_utils import MappingPacket

plt.switch_backend('agg')  # Use Agg backend for matplotlib

class GUI:
    def __init__(self, q_map2vis: Queue):
        dpg.create_context()
        dpg.create_viewport(title='WildGS-SLAM', width=1200, height=800)

        self.q_map2vis = q_map2vis
        self.has_terminated = False

        self.image_size = (dpg.get_viewport_width() // 3, dpg.get_viewport_height() // 2)
        self.traj_size = (dpg.get_viewport_width() // 2, dpg.get_viewport_height() // 2)
        self.splat_size = (dpg.get_viewport_width() // 2, dpg.get_viewport_height() // 2)

        self.gt = np.zeros((self.image_size[0],self.image_size[1], 4), dtype=np.float32)
        self.rendered = np.zeros((self.image_size[0], self.image_size[1], 4), dtype=np.float32)
        self.uncertainty = np.zeros((self.image_size[0], self.image_size[1], 4), dtype=np.float32)
        self.traj = np.zeros((self.traj_size[0], self.traj_size[1], 4), dtype=np.float32)
        self.splat = np.zeros((self.splat_size[0], self.splat_size[1], 4), dtype=np.float32)
        with dpg.texture_registry():
            dpg.add_raw_texture(
                self.image_size[0], self.image_size[1], self.gt, format=dpg.mvFormat_Float_rgba, id="gt"
            )
            dpg.add_raw_texture(
                self.image_size[0], self.image_size[1], self.rendered, format=dpg.mvFormat_Float_rgba, id="rendered"
            )
            dpg.add_raw_texture(
                self.image_size[0], self.image_size[1], self.uncertainty, format=dpg.mvFormat_Float_rgba, id="uncertainty"
            )
            dpg.add_raw_texture(
                self.traj_size[0], self.traj_size[1], self.traj, format=dpg.mvFormat_Float_rgba, id="trajectory"
            )
            dpg.add_raw_texture(
                self.splat_size[0], self.splat_size[1], self.splat, format=dpg.mvFormat_Float_rgba, id="splat"
            )


        with dpg.window(label="Ground Truth Images", width=self.image_size[0], height=self.image_size[1], pos=(0, 0)):
            dpg.add_image("gt")
        with dpg.window(label="Rendered Images", width=self.image_size[0], height=self.image_size[1], pos=(self.image_size[0], 0)):
            dpg.add_image("rendered")
        with dpg.window(label="Uncertainty Images", width=self.image_size[0], height=self.image_size[1], pos=(self.image_size[0] * 2, 0)):
            dpg.add_image("uncertainty")
        with dpg.window(label="Trajectory", width=self.traj_size[0], height=self.traj_size[1], pos=(0, self.image_size[1])):
            dpg.add_image("trajectory")
        with dpg.window(label="Splat", width=self.splat_size[0], height=self.splat_size[1], pos=(self.traj_size[0], self.image_size[1])):
            dpg.add_image("splat")

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def update(self):
        if not self.q_map2vis.empty():
            data = self.q_map2vis.get()
            if isinstance(data, MappingPacket):
                self.gt = data.gt if data.gt is not None else self.gt
                self.rendered = data.rendered if data.rendered is not None else self.rendered
                self.uncertainty = data.uncertainty if data.uncertainty is not None else self.uncertainty
                self.traj = data.traj if data.traj is not None else self.traj
                self.splat = data.splat if data.splat is not None else self.splat
            elif isinstance(data, str) and data == "terminate":
                self.has_terminated = True

        dpg.set_value("gt", self.gt)
        dpg.set_value("rendered", self.rendered)
        dpg.set_value("uncertainty", self.uncertainty)
        dpg.set_value("trajectory", self.traj)
        dpg.set_value("splat", self.splat)

    def run(self):
        while dpg.is_dearpygui_running():
            self.update()
            dpg.render_dearpygui_frame()

            if self.has_terminated:
                break

        dpg.destroy_context()

def run(q_map2vis: Queue):
    gui = GUI(q_map2vis)
    gui.run()