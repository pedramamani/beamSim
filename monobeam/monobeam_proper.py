import proper
import matplotlib.pyplot as plt
import numpy as np
import math
from monobeam_config import *


class Beam:
    def __init__(self, wavelength, beam_width, profile=PROFILE.DEFAULT, grid_width=None,
                 grid_size=GRID_SIZE.DEFAULT):
        if not grid_width:
            grid_width = fx * beam_width
        self.wl = wavelength
        self.wf = proper.prop_begin(beam_width, wavelength, grid_size, beam_width / grid_width)
        self.grid_pos = [grid_width * i / grid_size for i in range(-grid_size // 2, grid_size // 2)]
        self.grid_size = grid_size

        if profile == PROFILE.UNIFORM:
            amplitude_dist = [
                [1 if (x ** 2 + y ** 2 <= beam_width ** 2 / 4) else 0 for x in self.grid_pos] for y in self.grid_pos]
        else:
            amplitude_dist = [
                [np.exp(-8 * (x ** 2 + y ** 2) / beam_width ** 2) for x in self.grid_pos] for y in self.grid_pos]

        proper.prop_add_phase(self.wf, [[0 for _ in self.grid_pos] for _ in self.grid_pos])
        proper.prop_multiply(self.wf, amplitude_dist)
        proper.prop_define_entrance(self.wf)

    def add_quadratic_phase(self, focal_length):
        proper.prop_lens(self.wf, focal_length)

    def propagate(self, distance):
        proper.prop_propagate(self.wf, distance, PHASE_OFFSET=True)

    def add_linear_phase(self, angle_x=0, angle_y=0):
        tangent_x = math.tan(angle_x)
        tangent_y = -math.tan(angle_y)
        phase_dist = [[(tangent_x * x + tangent_y * y) for x in self.grid_pos] for y in self.grid_pos]
        proper.prop_add_phase(self.wf, phase_dist)

    def pass_rect_aperture(self, width, height):
        proper.prop_rectangular_aperture(self.wf, width=width, height=height)

    def pass_circ_aperture(self, diameter):
        proper.prop_circular_aperture(self.wf, radius=diameter / 2)

    def plot_phase_profile(self, title=None):
        profile = proper.prop_get_phase(self.wf)[self.grid_size // 2]
        plt.plot(profile)
        plt.title(title)
        plt.ylim(-np.pi, np.pi)
        plt.show()

    def plot_amplitude_profile(self, title=None):
        profile = proper.prop_get_amplitude(self.wf)[self.grid_size // 2]
        plt.plot(profile)
        plt.title(title)
        plt.show()

    def get_phase_profile(self):
        return proper.prop_get_phase(self.wf)[self.grid_size // 2]

    def get_amplitude_profile(self):
        return proper.prop_get_amplitude(self.wf)[self.grid_size // 2]
