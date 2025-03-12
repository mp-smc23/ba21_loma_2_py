from common.enums import WindowType
from dsp.phase import *
from dsp.resample import *
from dsp.transform import *
from dsp.source import AudioSource
from dsp.wrapper import PitchShiftWrapper

if __name__ == '__main__':
    """
    Main function that starts all operations.
    set the track_info.half_tone_steps_to_shift to pitch shift a file, stored inside /res/test-data
   """
    import librosa
    import matplotlib.pyplot as plt
    import sounddevice as sd

    # new version
    track_info = TrackInfo()
    x, sr = librosa.load("test.wav", sr=None)
    track_info.sample_rate = sr
    track_info.hop_size_factor = 4
    track_info.normalize = False
    track_info.windowType = WindowType.hann.name
    track_info.half_tone_steps_to_shift = 5
    
    audio_source = AudioSource(track_info, x, sr)
    track = audio_source.get_track()
    
    wrapper = PitchShiftWrapper(PitchShifter(track.info,
                                             PhaseLockedDynamicShifter(track.info, magnitude_min_factor=10 ** -6), 
                                             LibrosaResampler(track.info)))
    
    transformed_samples = wrapper.process(track)
    
    # plot both spectrograms:
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.specgram(x, Fs=sr)
    plt.title('Original')
    plt.subplot(2, 1, 2)
    plt.specgram(transformed_samples, Fs=sr)
    plt.title('Transformed')
    plt.show()
    
    input("Press Enter to play the original audio")
    sd.play(x, sr)
    input("Press Enter to play the transformed audio")
    sd.play(transformed_samples, sr)
    input("Press Enter to exit")
    
    