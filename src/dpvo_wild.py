# Adapted from DPVO.src.dpvo.py

import subprocess
import traceback
import numpy as np

from dpvo.lietorch import SE3
from dpvo.net import VONet
from dpvo.patchgraph import PatchGraph
from src.utils.datasets import load_metric_depth
from src.utils.mono_priors.img_feature_extractors import predict_img_features
from src.utils.mono_priors.metric_depth_estimators import predict_metric_depth
from dpvo import altcorr, fastba, lietorch
from dpvo import projective_ops as pops
from dpvo.utils import *

from collections import deque

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


def unnormalize(tensor):
    # input tensor is in (2*x - 0.5) form; invert: x = (tensor + 0.5) / 2
    out = (tensor + 0.5) / 2
    return out.clamp(0, 1)


def fully_unload_model(model):
    # Push tensors to CPU
    for param in model.parameters():
        param.data = param.data.cpu()
        if param._grad is not None:
            param._grad.data = param._grad.data.cpu()
    for buffer in model.buffers():
        buffer.data = buffer.data.cpu()

    # Optional: make sure model doesn’t autoload again
    model.eval()

    # Clean CUDA allocator
    torch.cuda.empty_cache()



GPU_COLUMNS = [
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total"
]


def get_full_gpu_stats():
    result = subprocess.check_output(
        ['nvidia-smi',
         f'--query-gpu={",".join(GPU_COLUMNS)}',
         '--format=csv,nounits,noheader']
    ).decode('utf-8')

    stats = result.strip().split('\n')
    gpu_info = []
    assert len(stats) == 1, 'need to redo logic'
    values = list(map(int, stats[0].split(', ')))
    return values


autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO_Wild:
    def __init__(self, cfg, network, video, ht=480, wd=640, viz=False, metric_depth_estimator=None,
                 feat_extractor=None):
        self.video = video
        self.video_buffer = RingBuffer(cfg['tracking']['buffer'])
        self.cfg = cfg['dpvo']
        self.cfg_slam = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False

        # torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht  # image height
        self.wd = wd  # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        self.mono_disps = torch.zeros(self.N, ht, wd, device='cpu')
        self.mono_disps_estimated = np.zeros(self.N)
        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        if viz:
            self.start_viewer()
        self.metric_depth_estimator = metric_depth_estimator
        self.feat_extractor = feat_extractor

    def load_long_term_loop_closure(self):
        try:
            from dpvo.loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v

            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        # print(f'TERMINATE {datetime.datetime.now()}')
        # print(get_full_gpu_stats())
        # fully_unload_model(self.metric_depth_estimator)
        # fully_unload_model(self.feat_extractor)
        # print(f'TERMINATE {datetime.datetime.now()}')
        # print(get_full_gpu_stats())
        # if self.cfg.CLASSIC_LOOP_CLOSURE:
        #     self.long_term_lc.terminate(self.n)
        #
        # if self.cfg.LOOP_CLOSURE:
        #     self.append_factors(*self.pg.edges_loop())
        #
        # for _ in range(12):
        #     self.ran_global_ba[self.n] = False
        #     self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        if self.cfg.BETA_WEIGHT:
            P = self.pg.patches_.cpu()
            kk = full_kk.cpu()
            ix = self.pg.ix.cpu()
            kx = torch.unique(kk)
            xy = P[ix[kx], kx % self.M, :, self.P // 2, self.P // 2][:, :2]
            jj = full_ii.cpu()  # (E,)
            Hs, Ws = self.uncertainties_inv.shape[-2:]
            scale = self.RES / self.down_scale
            x_idx = (xy[:, 0] * scale).round().long().clamp(0, Ws - 1)
            y_idx = (xy[:, 1] * scale).round().long().clamp(0, Hs - 1)
            u_val = self.uncertainties_inv[jj, y_idx, x_idx]
            full_weight *= u_val[:, None]
        self.init_depths(full_kk)
        depth_prior  = self.get_depth_prior(full_kk)
        fastba.BA(self.poses, self.patches, self.intrinsics,
                  full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n,
                  M=self.M, iterations=2, eff_impl=True, alpha1=self.cfg.alpha1, alpha2=self.cfg.alpha2,
                  c_depth_reg=self.cfg.c_depth_reg,
                  depth_prior=depth_prior)
        self.ran_global_ba[self.n] = True


    def init_depth(self, ts):
        if self.mono_disps_estimated[ts] > 0:
            return

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            intrinsics, image_input = self.video_buffer[ts]
            print('estimating metric depth dpvo', ts)
            save_dir = self.cfg_slam["data"]["output"] + "/" + self.cfg_slam["scene"]
            try:
                mono_depth_u = load_metric_depth(ts, save_dir).to('cuda')
            except Exception as e:
                mono_depth_u = predict_metric_depth(self.metric_depth_estimator, ts, image_input, self.cfg_slam, 'cuda')


            slice_h = slice(self.RES // 2 - 1, self.ht // self.RES * self.RES + 1, self.RES)
            slice_w = slice(self.RES // 2 - 1, self.wd // self.RES * self.RES + 1, self.RES)

            mono_depth = mono_depth_u[slice_h, slice_w]
            self.mono_disps[ts] = torch.where(mono_depth > 0, 1 / mono_depth, 0).cpu()
            dino_features = predict_img_features(self.feat_extractor, ts, image_input, self.cfg_slam, 'cuda')

            # self.video.append(ts, image_input[0], SE3.Identity(1,).data.squeeze() if ts==0 else None, 1.0 if ts ==0 else None, mono_depth_u, intrinsics / float(self.video.down_scale), None, None, None, dino_features)
        #

        self.mono_disps_estimated[ts] = 1

    def init_depths(self, kk):
        if self.cfg.alpha1 == 0:
            return
        kk = kk.cpu()
        ix = self.pg.ix.cpu()
        kx = torch.unique(kk)
        ts_s = np.unique(self.pg.tstamps_[ix[kx]])
        max_ts_threshold = np.max(ts_s) - self.cfg.KEYFRAME_THRESH + 1
        first_ba = self.mono_disps_estimated[0] == 0

        if not first_ba:
            ts_s = [int(ts) for ts in ts_s]
        else:
            ts_s = [int(ts) for ts in ts_s if ts < max_ts_threshold]

        for ts in ts_s:
            self.init_depth(ts)
        if first_ba:
            print('INITIALIZE PATCHES')
            # xy = patches[0, :, self.P // 2, self.P // 2, :2]
            # xx, yy = xy.unbind(dim=1)
            # depth = self.mono_disps[tstamp, yy.cpu().numpy(), xx.cpu().numpy()]
            # depths = self.get_depth_prior(kk)
            # for n, ts in enumerate(ts_s):
            #     xy = self.pg.patches_[n, :, self.P // 2, self.P // 2, :2]
            #     xx, yy = xy.unbind(dim=1)
            #     depth = self.mono_disps[ts, yy.cpu().numpy(), xx.cpu().numpy()]
            #     self.pg.patches_[n, :, :, 2] = (torch.as_tensor(depth, dtype=torch.float32)
            #                                     .to(self.pg.patches_.device)
            #                                     .view(1,self.pg.patches_.shape[1],1, 1)
            #                                     )
            #     self.pg.patches_[n, :, :, 2] = torch.as_tensor(depths[n], dtype=torch.float32).to(self.pg.patches_.device).view(1, self.pg.patches_.shape[2], 1, 1)


    def get_depth_prior(self, kk):
        if self.cfg.alpha1 == 0:
            return
        P = self.pg.patches_.cpu()
        kk = kk.cpu()
        ix = self.pg.ix.cpu()
        kx = torch.unique(kk)
        xy = P[ix[kx], kx % self.M, :, self.P // 2, self.P // 2][:, :2]
        xx, yy = xy.unbind(dim=1)
        dd = self.pg.tstamps_[ix[kx]]
        print('unique dd', np.unique(dd))
        depth_prior = self.mono_disps[dd, yy.numpy(), xx.numpy()].to(self.patches.device).clone()
        depth_prior[dd > dd.max() - 4] = 0
        return depth_prior

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[..., self.P // 2, self.P // 2] + delta.float()
            if self.cfg.BETA_WEIGHT:
                P = self.pg.patches_.cpu()
                ii = self.pg.ii.cpu()
                kk = self.pg.kk.cpu()
                ix = self.pg.ix.cpu()
                kx = torch.unique(kk)
                xy = P[ix[kx], kx % self.M, :, self.P // 2, self.P // 2][:, :2]
                jj = self.pg.ii.cpu()  # (E,)
                Hs, Ws = self.video.uncertainties_inv.shape[-2:]
                scale = self.RES / self.video.down_scale
                x_idx = (xy[:, 0] * scale).round().long().clamp(0, Ws - 1)
                y_idx = (xy[:, 1] * scale).round().long().clamp(0, Hs - 1)
                u_val = self.video.uncertainties_inv[jj, y_idx, x_idx]
                weight *= u_val[:, None]

        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    self.init_depths(self.pg.kk)
                    depth_prior = self.get_depth_prior(self.pg.kk)
                    fastba.BA(self.poses, self.patches, self.intrinsics,
                              target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n,
                              M=self.M, iterations=2, eff_impl=False, alpha1=self.cfg.alpha1, alpha2=self.cfg.alpha2,
                              c_depth_reg=self.cfg.c_depth_reg, depth_prior=depth_prior)
            except Exception as e:
                print(f"Warning BA failed...{e}")
                traceback.print_exc()

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
                            torch.arange(max(self.n - r, 0), self.n, device="cuda"), indexing='ij')

    def _edges_forw(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n - 1, self.n, device="cuda"), indexing='ij')

    def _edges_back(self):
        r = self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
                            torch.arange(max(self.n - r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics, video=None, save=True):
        print(f'[dpvo] {int(tstamp)=} {[int(i) for i in self.pg.tstamps_ if i > 0]}')
        """ track new frame """
        print('GPU_STATS')
        print(get_full_gpu_stats())
        global_bundle_adjustment = False
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n + 1) >= self.N:
            raise Exception(
                f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N * 2}"')

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())
        # adjustment for WILD
        if video is not None:
            image_input = image
            img_bgr = image[:, [2, 1, 0], :, :]
            image = (2 * img_bgr[None, :] - 0.5).to('cuda')
        else:
            image_input = image
            image = 2 * (image[None, None] / 255.0) - 0.5

        self.video_buffer.append((intrinsics,image_input.cpu()))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT,
                    return_color=True)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.pg.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1
        if self.n > 0 and not self.is_initialized:
            print('motion_probe', self.motion_probe().detach().cpu().numpy())
            if self.motion_probe() < 2:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)
                    global_bundle_adjustment = True

        # Add forward and backward factors
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 12 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()
        return global_bundle_adjustment
