import abc
import heapq
from dataclasses import dataclass
from typing import List

import numpy as np

from ps.pv_dr.common import utils
from ps.pv_dr.common.avg_queue import AvgQueue
from ps.pv_dr.common.enums import TransientDetectionType, PhaseResetType
from ps.pv_dr.common.movingmedian import MovingMedian
from ps.pv_dr.common.track import TrackInfo

class PhaseShifter:
    """Base class for the various phase shifter algorithms"""

    def __init__(self, info: TrackInfo):
        """
        Initializes the base class
        Stores the track info locally and calculates the expected phase delta for the hop and frame size
        :param info: A TrackInfo object providing information about the track and the transformation
        """
        self.rising_count = 0
        self.info = info
        self.last_magnitude = np.zeros(self.info.frame_size_nyquist)
        self.phase_delta_target = (2.0 * np.pi * self.info.hop_size_analysis) * np.array(range(0, self.info.frame_size_nyquist)) / self.info.frame_size_padded
        self.phase_analysis_prev = np.zeros(self.info.frame_size_nyquist)
        self.phase_synthesis = np.zeros(self.info.frame_size_nyquist)
        self.transient_prob_threshold = 0.35
        self.transient_prob_prev = 0
        self.high_freq_mag_sum_last = 0
        self.high_freq_filter = MovingMedian(19, 85)
        self.high_freq_deriv_filter = MovingMedian(19, 90)
        self.high_freq_deriv_delta = 0
        self.last_high_freq_deriv_delta = 0
        self.max_mag_avg_queue = AvgQueue(19)
        # amplitude (root-power) ratio equivalent to 3 dB (10**(3/20)=10**0.15)
        # mag1 / mag2 = 10**0.15 ---> mag1 is 3 dB over mag2
        self.magnitude_ratio_3db = 10 ** 0.15
        self.transient_magnitude_min_factor = 10e-6

        self.band_low = int(np.floor(150 * info.frame_size_padded / info.sample_rate))
        self.band_high = int(np.floor(1000 * info.frame_size_padded / info.sample_rate))
        self.mid_range = slice(0, self.info.frame_size_nyquist)

        self.frame_index = 0

    @abc.abstractmethod
    def process(self, frame_fft: List[complex]) -> List[float]:
        return []

    def high_frequency_transient_detection(self, magnitude):
        """"
        Sums the magnitudes of all bins weighted by their frequencies and puts the results in a moving median. If the median is rising three subsequent frames a transient is detected.
        :param magnitude: magnitude of the current frame
        :return: high frequency transient probability in range from 0 to 1
        """
        high_freq_mag_sum = 0.0
        transient_probability = 0.0
        # sum the magnitudes of all bins weighted by their center frequency
        for n in range(0, len(magnitude)):
            high_freq_mag_sum = high_freq_mag_sum + magnitude[n] * n

        high_freq_deriv = high_freq_mag_sum - self.high_freq_mag_sum_last
        self.high_freq_filter.put(high_freq_mag_sum)
        self.high_freq_deriv_filter.put(high_freq_deriv)
        high_freq_filtered = self.high_freq_filter.get()
        high_freq_deriv_filtered = self.high_freq_deriv_filter.get()
        self.high_freq_mag_sum_last = high_freq_mag_sum
        high_freq_deriv_delta = 0.0
        high_freq_excess = high_freq_mag_sum - high_freq_filtered

        if high_freq_excess > 0:
            # if current high frequency content is above the current median, the difference in gradiant to the median is calculated
            high_freq_deriv_delta = high_freq_deriv - high_freq_deriv_filtered

        if high_freq_deriv_delta < self.last_high_freq_deriv_delta:
            # if the current difference in gradient is smaller than that of the last frame the rising count is reset
            if self.rising_count > 3 and self.last_high_freq_deriv_delta > 0:
                # if the difference in gradient from the median has been rising for at least 3 frames, the current frame holds a transient
                transient_probability = 0.5
            self.rising_count = 0
        else:
            # while the difference in gradient is larger than that of the previous frame, we're on the rising slope before the transient
            self.rising_count = self.rising_count + 1
        self.last_high_freq_deriv_delta = high_freq_deriv_delta
        return transient_probability

    def percussive_transient_detection(self, magnitude):
        """"
        Calculates the percussive transient probability by counting all the significant and the non zero bins.
        :param magnitude: magnitude of the current frame
        :return: percussive transient probability in range from 0 to 1
        """
        self.max_mag_avg_queue.push_pop(max(magnitude))
        zeroThresh = self.transient_magnitude_min_factor * self.max_mag_avg_queue.get_avg()

        count = 0
        nonZeroCount = 0

        for n in range(0, len(magnitude)):
            magnitude_increase_ratio = 0.0
            if self.last_magnitude[n] > zeroThresh:
                # calculate magnitude growth since last frame if last magnitude of current bin is non-zero
                magnitude_increase_ratio = magnitude[n] / self.last_magnitude[n]
            elif magnitude[n] > zeroThresh:
                # if last magnitude of current bin is below zero threshold but current magnitude is significant, default to 3dB ratio
                magnitude_increase_ratio = self.magnitude_ratio_3db
            # count significant magnitude increases
            if magnitude_increase_ratio >= self.magnitude_ratio_3db: count += 1
            # count significant magnitudes
            if magnitude[n] > zeroThresh: nonZeroCount += 1

        self.last_magnitude = magnitude
        if (nonZeroCount == 0):
            return 0
        # return the ratio of bins with significant magnitude and bins with significant magnitude which translates to the likelihood of the current frame being a transient
        # count is always smaller than nonZeroCount. The returned ratio therefore has the range [0, 1]
        # small difference between count and nonZeroCount indicate few significant bins without significant growth -> likely a transient
        return count / nonZeroCount

    def transient_detection(self, magnitude):
        """"
        Calculates if a transient is detected by comparing the current transient likelihood with the last likelihood and a likelihood threshold.
        :param magnitude: magnitude of the current frame
        :return: True if transient probability indicates transient
        """
        transient_prob = max(self.percussive_transient_detection(magnitude), self.high_frequency_transient_detection(magnitude))
        if transient_prob > 0 and transient_prob > self.transient_prob_prev and transient_prob > self.transient_prob_threshold:
            self.transient_prob_prev = transient_prob
            return True
        self.transient_prob_prev = transient_prob
        return False

    def phase_reset(self, phase):
        """"
        Resets the phases and sets the current mid range
        :param phase: untransformed phase of the current frame
        :return: returns the default mid_range
        """
        self.phase_synthesis[0: self.band_low] = phase[0: self.band_low]
        self.phase_synthesis[self.band_high: self.info.frame_size_nyquist] = phase[self.band_high: self.info.frame_size_nyquist]
        return slice(self.band_low, self.band_high)


class PhaseLockedDynamicShifter(PhaseShifter):
    """Based on Phase Vocoder Done Right pseudo code https://www.researchgate.net/publication/319503719_Phase_Vocoder_Done_Right"""

    def __init__(self, info: TrackInfo, transient_detection_mode: TransientDetectionType = TransientDetectionType.NONE,
                 phase_reset_type: PhaseResetType = PhaseResetType.FULL_RANGE, magnitude_min_factor=10 ** -6):
        super().__init__(info)

        self.magnitude_min_factor = magnitude_min_factor
        self.max_magnitude = 0
        self.magnitude_prev = np.zeros(self.info.frame_size_nyquist)
        self.phase_delta_prev = np.zeros(self.info.frame_size_nyquist)

    def process(self, frame_fft: List[complex]) -> List[float]:
        """
        Calculates the expected phase shift using the significant magnitudes from the current and last frame and placing them into a self sorting max heap.
        Calculating the partial derivatives the phase influenced by the magnitudes in the heap are then propagated in a vertical direction.
        :param frame_fft: a frame in the frequency domain spectrum
        :return: the transformed phase for each bin
        """
        magnitude = abs(frame_fft)
        # get imaginary values from fft
        phase_analysis = np.angle(frame_fft)

        # calculate the diff between last and current phase vector)
        phase_delta = self.phase_delta_target + utils.princarg(phase_analysis - self.phase_analysis_prev - self.phase_delta_target)
        phase_delta = phase_delta * self.info.time_stretch_ratio

        mid_range = self.mid_range
        transient_detected = self.transient_detection(magnitude)
        if transient_detected:
            mid_range = self.phase_reset(phase_analysis)


        self.max_magnitude = max(max(magnitude), self.max_magnitude)
        min_magnitude = self.magnitude_min_factor * self.max_magnitude

        significant_magnitudes = {i: magnitude[i] for i in range(0, self.info.frame_size_nyquist)[mid_range] if magnitude[i] > min_magnitude}
        max_heap = [HeapBin(i, -1, self.magnitude_prev[i], 0) for i in significant_magnitudes.keys()]
        heapq.heapify(max_heap)

        # perform simple horizontal phase propagation for bins with insignificant magnitude
        for i in range(0, self.info.frame_size_nyquist)[mid_range]:
            if i not in significant_magnitudes.keys():
                self.phase_synthesis[i] = self.phase_synthesis[i] + phase_delta[i]

        while len(significant_magnitudes) > 0 and len(max_heap) > 0:
            max_bin = heapq.heappop(max_heap)
            bin_index = max_bin.bin_index
            time_index = max_bin.time_index
            if time_index < 0 and bin_index in significant_magnitudes.keys():
                # bin has been taken from the last frame and horizontal phase propagation (using backwards phase time derivative and trapezoidal integration) is needed
                self.phase_synthesis[bin_index] = self.phase_synthesis[bin_index] + (self.phase_delta_prev[bin_index] + phase_delta[bin_index]) / 2
                # add the current bin of the current frame to the heap for further processing
                heapq.heappush(max_heap, HeapBin(bin_index, 0, significant_magnitudes.get(bin_index), utils.princarg(self.phase_synthesis[bin_index] - phase_analysis[bin_index])))
                # remove the processed bin from the set
                significant_magnitudes.pop(bin_index)

            if time_index >= 0:
                # the bin is from the current frame and vertical phase propagation (potentially in both directions) is needed
                for bin_index_other in (bin_index - 1, bin_index + 1):
                    # check if the surrounding two bins have significant magnitudes
                    if bin_index_other in significant_magnitudes.keys():
                        self.phase_synthesis[bin_index_other] = phase_analysis[bin_index_other] + max_bin.phase_rotation
                        # add the next / prev bin to the heap for further processing
                        heapq.heappush(max_heap, HeapBin(bin_index_other, 0, magnitude[bin_index_other], max_bin.phase_rotation))
                        # remove the processed bin from the set
                        significant_magnitudes.pop(bin_index_other)

        self.phase_analysis_prev = np.copy(phase_analysis)
        self.phase_delta_prev = np.copy(phase_delta)
        self.magnitude_prev = np.copy(magnitude)
        self.frame_index += 1

        return self.phase_synthesis


@dataclass
class HeapBin:
    bin_index: int
    time_index: int
    magnitude: float
    phase_rotation: float

    def __lt__(self, other): return self.magnitude > other.magnitude

    def __eq__(self, other): return self.magnitude == other.magnitude

    def __str__(self): return str(self.magnitude)
