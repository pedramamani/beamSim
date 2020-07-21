from polybeam import PolyBeam
import numpy as np


def routine_4f_shaper(M):
    beam = PolyBeam(ν0=375, Δν=10, Δx=3)

    beam.plot_amplitude_freq(title='Input Amplitude').plot_phase_freq(title='Input Phase')
    beam.plot_amplitude_time(title='Input Amplitude').plot_phase_time(title='Input Phase')

    # beam.rotate(α=3)
    # beam.chirp(α=10E6)
    # beam.disperse(d=1300, α=70)

    # beam.propagate(Δz=30).lens(f=30).propagate(Δz=30)
    # beam.mask(M=M)
    # beam.propagate(Δz=30).lens(f=30).propagate(Δz=30)
    # beam.disperse(d=1300, α=70)  # negative dispersion to collimate beam
    # todo: subtract central frequency instead for phase
    # todo: center around the holistic maximum for transform
    # todo: fix mask and rotate for non-planar

    # beam.plot_amplitude_freq(title='Output Amplitude')  # .plot_phase_freq(title='Output Phase')
    # beam.plot_amplitude_time(title='Output Amplitude')  # .plot_phase_time(title='Output Phase')
    # beam.plot_wigner()


if __name__ == '__main__':
    routine_4f_shaper(lambda xs: np.where(np.abs(xs) < 2, 0, 1))  # sharp block
    # routine_4f_shaper(lambda xs: np.where(np.abs(xs) < 2, 1, 0))  # sharp slit
