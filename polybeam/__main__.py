from polybeam import PolyBeam
import numpy as np


def routine_4f_shaper(M):
    beam = PolyBeam(ν0=3.75E14, Δν=1E13, Δx=3E-3)

    beam.plot_amplitude_time(title='Input Amplitude').plot_phase_time(title='Input Phase')
    beam.disperse(d=1E-3 / 1300, α=np.deg2rad(70))
    beam.plot_amplitude_freq(title='Input Amplitude').plot_phase_freq(title='Input Phase')

    # beam.chirp(α=1000)
    beam.propagate(Δz=0.3).lens(f=0.3).propagate(Δz=0.3)
    beam.mask(M=M)
    beam.propagate(Δz=0.3).lens(f=0.3).propagate(Δz=0.3)
    beam.disperse(d=1E-3 / 1300, α=np.deg2rad(70))  # negative dispersion to collimate beam

    beam.plot_amplitude_freq(title='Output Amplitude').plot_phase_freq(title='Output Phase')
    beam.plot_amplitude_time(title='Output Amplitude').plot_phase_time(title='Output Phase')


if __name__ == '__main__':
    # routine_4f_shaper(lambda x: np.where(np.abs(x) < 3E-3, 1, 0))  # sharp mask
    # routine_4f_shaper(lambda x: np.exp(-(x / 1E-3) ** 2))  # soft mask
    routine_4f_shaper(lambda x: np.where(np.abs(x) > 2E-3, 0, 1))  # sharp slit
