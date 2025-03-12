import numpy as np

from ps.pv_dr.common.track import Track, TrackInfo


class AudioSource:
    def __init__(self, info: TrackInfo, x: np.array, sr: int):
        self.track = Track()
        
        self.track.info = info
        
        self.track.base = x
        self.track.info.sample_rate = sr

        self.track.info.setup()

    def get_track(self) -> Track:
        return self.track
