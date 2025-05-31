import matplotlib.pyplot as plt
import numpy as np
import dearpygui.dearpygui as dpg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from evo.tools import plot
from evo.core.trajectory import PoseTrajectory3D
from multiprocessing import Queue
from src.new_gui.gui_utils import MappingPacket
import time

plt.switch_backend('agg')  # Use Agg backend for matplotlib

class GUI:
    def __init__(self, q_map2vis: Queue):

        IMAGE_SIZE = (640, 480)  # Default image size
        IMAGE_SIZE = (int(IMAGE_SIZE[0] * 0.75), int(IMAGE_SIZE[1] * 0.75))  # Resize to 75% of original size
        PADDING = 5  # Padding around the images

        dpg.create_context()
        dpg.create_viewport(title='WildGS-SLAM', width=IMAGE_SIZE[0]*3, height=900)

        self.q_map2vis = q_map2vis
        self.has_terminated = False

        self.traj = []
        self.previous_update_time = time.time()
        self.frame_timestamps = []

        self.metrics_size = (dpg.get_viewport_width(), 50)
        self.image_size = IMAGE_SIZE
        self.traj_size = (dpg.get_viewport_width() // 2, dpg.get_viewport_height() // 2)
        self.splat_size = (640, 480)

        self.gt = np.zeros((self.image_size[0],self.image_size[1], 4), dtype=np.float32)
        self.rendered = np.zeros((self.image_size[0], self.image_size[1], 4), dtype=np.float32)
        self.uncertainty = np.zeros((self.image_size[0], self.image_size[1], 4), dtype=np.float32)
        self.traj_img = np.zeros((self.traj_size[0], self.traj_size[1], 4), dtype=np.float32)
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
                self.traj_size[0], self.traj_size[1], self.traj_img, format=dpg.mvFormat_Float_rgba, id="trajectory"
            )
            dpg.add_raw_texture(
                self.splat_size[0], self.splat_size[1], self.splat, format=dpg.mvFormat_Float_rgba, id="splat"
            )

        with dpg.window(label="Metrics", width=self.metrics_size[0], height=self.metrics_size[1], pos=(0, 0)):
            with dpg.group(horizontal=True):
                dpg.add_text("FPS:", )
                dpg.add_text("0", tag="fps", color=(255, 0, 0, 255)) 
                dpg.add_spacer(width=20) 
                dpg.add_text("Time since last update:")
                dpg.add_text("0.0", tag="last_update", color=(255, 0, 0, 255))   
        with dpg.window(label="Ground Truth Images", width=self.image_size[0]+PADDING, height=self.image_size[1]+PADDING, pos=(0, self.metrics_size[1])):
            dpg.add_image("gt")
        with dpg.window(label="Rendered Images", width=self.image_size[0]+PADDING, height=self.image_size[1]+PADDING, pos=(self.image_size[0]+PADDING, self.metrics_size[1])):
            dpg.add_image("rendered")
        with dpg.window(label="Uncertainty Images", width=self.image_size[0]+PADDING, height=self.image_size[1]+PADDING, pos=(self.image_size[0]*2+PADDING*2, self.metrics_size[1])):
            dpg.add_image("uncertainty")
        with dpg.window(label="Trajectory", width=self.traj_size[0], height=self.traj_size[1], pos=(0, self.image_size[1]+self.metrics_size[1])):
            dpg.add_image("trajectory")
        with dpg.window(label="Splat", width=self.splat_size[0], height=self.splat_size[1], pos=(self.traj_size[0], self.image_size[1]+self.metrics_size[1])):
            dpg.add_image("splat")

        dpg.setup_dearpygui()
        dpg.show_viewport()

        self.q_map2vis.get()

    def update_traj(self, new_traj):
        self.traj.append(new_traj)
        
        traj = PoseTrajectory3D(poses_se3=np.array(self.traj), timestamps=[0, 1])
        fig = plt.figure(figsize=(self.traj_size[0] / 100, self.traj_size[1] / 100), dpi=100)
        canvas = FigureCanvasAgg(fig)
        plot_mode = plot.PlotMode.xyz
        ax = plot.prepare_axis(fig, plot_mode, length_unit=Unit.meters)
        plot.traj(ax, plot_mode, traj)
        plot.draw_coordinate_axes(ax, traj, plot_mode, 0.01)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        image = image.astype(np.float32) / 255

        image = cv2.resize(image, (self.traj_size[0], self.traj_size[1]))

        self.traj_img = image

    def update(self):
        if not self.q_map2vis.empty():
            print("Updating GUI...")
            data = self.q_map2vis.get()
            if isinstance(data, MappingPacket):
                dpg.set_value("last_update", f"{time.time() - self.previous_update_time:.2f}")
                self.previous_update_time = time.time()
                if data.gt is not None:
                    # Resize to width
                    self.gt = cv2.resize(data.gt, (self.image_size[0], self.image_size[1]))
                    self.gt = cv2.cvtColor(self.gt, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
                if data.rendered is not None:
                    self.rendered = cv2.resize(data.rendered, (self.image_size[0], self.image_size[1]))
                    self.rendered = cv2.cvtColor(self.rendered, cv2.COLOR_BGR2RGBA).astype(np.float32)
                if data.uncertainty is not None:
                    self.uncertainty = cv2.resize(data.uncertainty, (self.image_size[0], self.image_size[1]))
                    self.uncertainty = cv2.cvtColor(self.uncertainty, cv2.COLOR_BGR2RGBA).astype(np.float32)
                if data.splat is not None: 
                    self.splat = cv2.resize(data.splat, (self.splat_size[0], self.splat_size[1]))
                    self.splat = cv2.cvtColor(self.splat, cv2.COLOR_BGR2RGBA).astype(np.float32)

                if data.traj is not None:
                    self.update_traj(data.traj)

                if data.gt is not None:
                    self.frame_timestamps.append(time.time())
                    if len(self.frame_timestamps) > 30:
                        self.frame_timestamps.pop(0)
                    if len(self.frame_timestamps) > 2:
                        fps = 1 / np.diff(self.frame_timestamps).mean()
                        dpg.set_value("fps", f"{fps:.2f}")


            elif isinstance(data, str) and data == "terminate":
                self.has_terminated = True

            dpg.set_value("gt", self.gt)
            dpg.set_value("rendered", self.rendered)
            dpg.set_value("uncertainty", self.uncertainty)
            dpg.set_value("trajectory", self.traj_img)
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