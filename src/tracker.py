from collections import deque

from src.dpvo_wild import DPVO_Wild
from src.utils.mono_priors.img_feature_extractors import predict_img_features
from src.utils.mono_priors.metric_depth_estimators import predict_metric_depth, get_metric_depth_estimator
import torch


def _project_patch_center(patch_xyz, K):
    """
    Project the 3‑D centre of a 3×3 patch to pixel coordinates.
    patch_xyz : torch.Tensor shape (3,)  — [x, y, z] (unknown scale)
    K         : torch/np 3×3            — camera intrinsics (fx, fy, cx, cy)
    Returns (u, v) as Python ints, or None if z <= 1e‑6.
    """
    x, y, z = patch_xyz.tolist()
    if z <= 1e-6:
        return None
    fx, fy, cx, cy = K
    u = int(torch.round(fx * x / z + cx))
    v = int(torch.round(fy * y / z + cy))
    return (u, v)

class RingBuffer:
    def __init__(self, max_size):
        self._buf = deque(maxlen=max_size)
        self._offset = 0               # index of the first element in _buf

    def append(self, item):
        if len(self._buf) == self._buf.maxlen:
            self._offset += 1          # one item is about to fall off
        self._buf.append(item)

    def __getitem__(self, idx):
        if idx < self._offset or idx >= self._offset + len(self._buf):
            raise IndexError("item has been evicted")
        return self._buf[idx - self._offset]

    def __len__(self):
        return len(self._buf)


# def _c2w_to_w2c(pose_vec7: torch.Tensor) -> torch.Tensor:
#     """
#     DPVO gives a 7‑vector [tx ty tz qx qy qz qw] in *camera→world*.
#     DepthVideo expects the inverse (*world→camera*).
#     """
#     return SE3(pose_vec7[None]).inv().data.squeeze()
def _c2w_to_w2c(poses: torch.Tensor) -> torch.Tensor:
    """
    DPVO gives a 7‑vector [tx ty tz qx qy qz qw] in *camera→world*.
    DepthVideo expects the inverse (*world→camera*).
    """
    return SE3(poses).data

def unnormalize(tensor):
    # input tensor is in (2*x - 0.5) form; invert: x = (tensor + 0.5) / 2
    out = (tensor + 0.5) / 2
    return out.clamp(0, 1)
import cv2
from yacs.config import CfgNode
from lietorch import SE3
import torch.nn.functional as F

from src.csv_profiler import CsvProfiler
import torch
from multiprocessing.connection import Connection
from src.utils.datasets import BaseDataset, load_metric_depth
from src.utils.Printer import Printer, FontColor
import numpy as np





class Tracker:
    def __init__(self, slam, pipe: Connection):
        self.profiler = CsvProfiler('tracker')
        self.cfg = slam.cfg
        self.device = self.cfg['device']
        self.net = None
        self.video = slam.video
        self.verbose = slam.verbose
        self.pipe = pipe
        self.output = slam.save_dir

        self.frontend_window = self.cfg['tracking']['frontend']['window']
        self.metric_depth_estimator = get_metric_depth_estimator(self.cfg)
        self.feat_extractor = slam.feat_extractor
        self.ba_freq = self.cfg['tracking']['backend']['ba_freq']
        self.printer: Printer = slam.printer

        self.cfg['dpvo'] = CfgNode(self.cfg['dpvo'])
        self.dpvo = DPVO_Wild(self.cfg,network='pretrained/dpvo.pth',
                         ht=384, wd=512,
                         viz=False,
                         metric_depth_estimator=self.metric_depth_estimator,
                         feat_extractor=self.feat_extractor,
                         video=self.video)
        # self.video.set_dpvo(self.dpvo)

    def _update_video(self, stream_buffer, intrinsic, update_dirty=False, is_init=False):
        # Prepare dpvo to get posese
        self.dpvo.traj = {}
        for i in range(self.dpvo.n):
            self.dpvo.traj[self.dpvo.pg.tstamps_[i]] = self.dpvo.pg.poses_[i]
        # assert np.max(self.dpvo.pg.tstamps_[:self.dpvo.n]) >=ids[12]

        # force_to_add_keyframe = (tstamp - last_tstamp) >= self.cfg['tracking']['force_keyframe_every_n_frames']
        last_tstamp = torch.max(self.video.timestamp)
        dpvo_timestamp = {t for t in self.dpvo.pg.tstamps_[:self.dpvo.n]}
        if is_init:

            ts_is = {int(ts) for ts in self.dpvo.pg.tstamps_}
            for ts_i in range(self.dpvo.pg.tstamps_[self.dpvo.n-1]):

                force_to_add_keyframe = (ts_i - last_tstamp) >= self.cfg['tracking']['force_keyframe_every_n_frames']

                if int(ts_i) in ts_is and self.dpvo.mono_disps_estimated[ts_i] == 0:
                    print(f'ts_i not estimated yet {ts_i}')
                    continue

                if ts_i in dpvo_timestamp or force_to_add_keyframe:
                    timestamp, image, _, _ = stream_buffer[ts_i]
                    assert ts_i == timestamp

                    save_dir = self.cfg["data"]["output"] + "/" + self.cfg["scene"]
                    try:
                        mono_depth = load_metric_depth(timestamp, save_dir).to('cuda')
                    except Exception as e:
                        print(f'loading metric depth failed estimated metric depth for {ts_i}')
                        mono_depth = predict_metric_depth(self.metric_depth_estimator, timestamp, image,
                                                            self.cfg, 'cuda')


                    dino_features = predict_img_features(self.feat_extractor, timestamp, image, self.cfg, self.device)
                    self.video.append(timestamp,
                                      image[0],
                                      None,  # was None
                                      1.0 if i ==0 else None,
                                      mono_depth,
                                      intrinsic / float(self.video.down_scale),
                                      None, None, None,
                                      dino_features)
                    last_tstamp = ts_i
        # Add new poses to video
        M = torch.max(self.video.timestamp)
        M_dpvo = np.max(self.dpvo.pg.tstamps_[:self.dpvo.n])
        # for i, ts_i in enumerate(self.dpvo.pg.tstamps_[:self.dpvo.n]):
        for ts_i in range(self.dpvo.pg.tstamps_[self.dpvo.n-1]):
            if ts_i > M and ts_i <= M_dpvo and not is_init:
                force_to_add_keyframe = (ts_i - last_tstamp) >= self.cfg['tracking']['force_keyframe_every_n_frames']
                if ts_i in dpvo_timestamp or force_to_add_keyframe:
                    timestamp, image, _, _ = stream_buffer[ts_i]
                    mono_depth = predict_metric_depth(self.metric_depth_estimator, timestamp, image, self.cfg, self.device)
                    dino_features = predict_img_features(self.feat_extractor, timestamp, image, self.cfg, self.device)
                    self.video.append(timestamp,
                                      image[0],
                                      None,  # was None
                                      1.0 if i ==0 else None,
                                      mono_depth,
                                      intrinsic / float(self.video.down_scale),
                                      None, None, None,
                                      dino_features)
                    last_tstamp = ts_i

        # Updating poses that are already in the video
        median_depth_est = torch.median(self.video.mono_disps_up[:self.video.counter.value].reshape(-1,1))
        median_depth_patch = torch.median(self.dpvo.pg.patches_[:self.dpvo.n].permute(0,1, 3, 4, 2).reshape(-1, 3)[:,2])

        scale_target = float(median_depth_est) / (float(median_depth_patch) + 1e-8)
        print(f'SCALE ADJUSTMENT {scale_target:1.3f} {median_depth_est=} {median_depth_patch=} {1/median_depth_patch=}')
        for i, ts_i in enumerate(self.video.timestamp):
            ts_i = int(ts_i)
            if ts_i ==0 and i != 0:
                break
            # This might interpolate the pose
            pose = self.dpvo.get_pose(ts_i).data.cpu().numpy()
            # pose[:3] *= 0.8
            if not  all(np.isclose(self.video.poses[i].cpu().numpy(), pose)) :
                converted = (
                    torch.from_numpy(pose)
                    .to(dtype=self.video.poses.dtype, device=self.video.poses.device)
                )
                self.video.poses[i] = converted

        if update_dirty:
            self.video.set_dirty(0, self.dpvo.n)



    def run(self, stream: BaseDataset):
        '''
        Trigger the tracking process.
        1. check whether there is enough motion between the current frame and last keyframe by motion_filter
        2. use frontend to do local bundle adjustment, to estimate camera pose and depth image, 
            also delete the current keyframe if it is too close to the previous keyframe after local BA.
        3. run online global BA periodically by backend
        4. send the estimated pose and depth to mapper, 
            and wait until the mapper finish its current mapping optimization.
        '''
        torch.manual_seed(1234)
        Id = SE3.Identity(1,).data.squeeze()

        prev_kf_idx = 0
        curr_kf_idx = 0
        prev_ba_idx = 0

        intrinsic = stream.get_intrinsic()
        stream_buffer = RingBuffer(max_size=self.cfg['tracking']['buffer'])

        for i in range(len(stream)):
            print(' ')
            print('self.video.counter.value',self.video.counter.value - 1)
            print('self.dpvo.counter ',self.dpvo.counter - 1)
            print('dpvo.n', self.dpvo.n)
            print('prev_kf_idx', prev_kf_idx)
            print('prev_ba_idx', prev_ba_idx)
            stream_buffer.append(stream[i])
            timestamp, image, _, _ = stream_buffer[i]

            with torch.no_grad():
                is_bundle_adjustment = self.dpvo(tstamp=timestamp, image=image, intrinsics=intrinsic, video=self.video)
                if self.dpvo.n == 0:
                    mono_depth = predict_metric_depth(self.metric_depth_estimator, timestamp, image, self.cfg,self.device)
                    dino_features = predict_img_features(self.feat_extractor, timestamp, image, self.cfg, self.device)
                    self.video.append(timestamp,
                                      image[0],  # CxHxW 3x384x512
                                      Id,
                                      1.0,
                                      mono_depth, #HxW 384x512
                                      intrinsic / float(self.video.down_scale), # vs 4 #TODO where is it used at all?
                                      None,#gmap,
                                      None,#net[0, 0],
                                      None,#inp[0, 0],
                                      dino_features) #27x36x384
            curr_kf_idx = self.dpvo.n - 1

            if curr_kf_idx != prev_kf_idx and self.dpvo.is_initialized:
                # TODO recheck logic
                if self.dpvo.n >= 16 and torch.max(self.video.timestamp)==0 : 
                    self._update_video(stream, intrinsic, is_init=True)
                    # assert self.video.counter.value == self.dpvo.n
                    self.pipe.send({"is_keyframe": True, "video_idx": len([t for t in self.video.timestamp if t !=0]),
                                        "timestamp": int(max([t for t in self.video.timestamp if t !=0])), "just_initialized": True,
                                        "end": False})
                    self.pipe.recv()
                    self.dpvo.down_scale = self.video.down_scale
                    # TODO Add second stage initialization for DPVO (self.video.uncertainties_inv)
                    # self.video.initialize_second_stage()
                elif torch.max(self.video.timestamp) >0:
                    if is_bundle_adjustment:
                        self.printer.print(f"BUNDLE ADJUSTMENT!!!!: {timestamp}",
                                       FontColor.TRACKER)
                        self.printer.print(f"Online BA at {curr_kf_idx}th keyframe, frame index: {timestamp} with n {self.dpvo.n}",
                                           FontColor.TRACKER)
                        self._update_video(stream, intrinsic,update_dirty=True)
                        # assert self.video.counter.value == self.dpvo.n
                        prev_ba_idx = curr_kf_idx
                    else:
                        self._update_video(stream, intrinsic)
                        # assert self.video.counter.value == self.dpvo.n
                    self.printer.print(f"Received continue from mapper, sending frame index: {timestamp}",
                                       FontColor.TRACKER)
                    self.pipe.send({"is_keyframe": True, "video_idx": len([t for t in self.video.timestamp if t !=0]),
                                    "timestamp": int(max([t for t in self.video.timestamp if t !=0])), "just_initialized": False,
                                    "end": False})
                    self.printer.print(f"Waiting for mapper to send continue, frame index: {timestamp}",
                                       FontColor.TRACKER)
                    self.pipe.recv()
            # TODO remove comment
            # if torch.max(self.video.timestamp) == 128:
            #     break
            prev_kf_idx = curr_kf_idx
            self.printer.update_pbar()
        self.printer.print('Tracker finished sending END-SIGNAL')
        self.pipe.send({"is_keyframe":True, "video_idx":None,
                        "timestamp":None, "just_initialized": False, 
                        "end":True})


                