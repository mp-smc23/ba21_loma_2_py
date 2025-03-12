import abc
from typing import List

import librosa
import numpy as np

from ps.pv_dr.common.track import TrackInfo


class Resampler:
    def __init__(self, info: TrackInfo):
        self.info = info

    @abc.abstractmethod
    def process(self, frame: List[float]) -> List[float]:
        return []

class LibrosaResampler(Resampler):

    def __init__(self, info: TrackInfo):
        super().__init__(info)

    def process(self, frame: List[float]) -> List[float]:
        """
        Resamples the given time domain frame using the prepared index and weight vectors
        :param frame: the time domain frame to be resamples
        :return: returns the resamples frame of the length info.frame_size_resampling
        """

        return librosa.resample(frame, orig_sr=self.info.frame_size, target_sr=self.info.frame_size_resampling)
