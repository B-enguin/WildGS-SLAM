import numpy as np

class MappingPacket:
    def __init__(
            self,
            gt = None,
            rendered = None,
            uncertainty = None,
    ):
        self.gt = gt
        self.rendered = rendered
        self.uncertainty = uncertainty