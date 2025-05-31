import numpy as np

class MappingPacket:
    def __init__(
            self,
            gt = None,
            rendered = None,
            uncertainty = None,
            splat=None,
            traj=None,
    ):
        self.gt = gt
        self.rendered = rendered
        self.uncertainty = uncertainty
        self.splat = splat
        self.traj = traj