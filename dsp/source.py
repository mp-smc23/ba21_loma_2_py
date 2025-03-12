import librosa
import numpy as np
import soundfile as sf

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


# class WavFileReader(AudioSource):
#     """Reads all data from a WAVE file and stores them into a Track."""
#     def __init__(self, info: TrackInfo, pitch_shift=True):
#         in_file = get_resources_root_test_data_path(info.name + ".wav")

#         track = Track()
#         track.info = info
#         data, track.info.sample_rate = sf.read(in_file, dtype='float32')
#         if len(np.shape(data)) > 1:
#             track.base = np.zeros(len(data))
#             channels = len(data[0])
#             for i in range(len(data)):
#                 track.base[i] = sum(data[i])
#             track.base /= channels
#         else:
#             track.base = data

#         track.info.setup()
        
#         super().__init__(track)
