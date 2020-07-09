from polybeam import PolyBeam
import numpy as np


def shaper_4f_routine():
    beam = PolyBeam(ν0=3.75E14, Δν=1E13, Δx=3E-3)

    # beam.plot_intensity_profile(title='Input Intensity')
    # beam.plot_phase_section(title='Input Phase')
    beam.plot_intensity_time(title='Input Intensity')
    beam.plot_phase_time(title='Input Phase')

    beam.disperse(d=1E-3 / 1300, α=np.deg2rad(70))
    # beam.rotate(α=np.deg2rad(3))
    # beam.chirp(1E-7 / 1E12) todo: what units for chirp?

    beam.propagate(Δz=0.3)
    beam.lens(f=0.3)
    beam.propagate(Δz=0.6)
    beam.lens(f=0.3)
    beam.propagate(Δz=0.3)
    beam.disperse(d=-1E-3 / 1300, α=-np.deg2rad(70))  # negative dispersion to collimate beam

    # beam.plot_intensity_profile(title='Output Intensity')
    # beam.plot_phase_profile(title='Output Phase')
    beam.plot_intensity_time(title='Output Intensity')
    beam.plot_phase_time(title='Output Phase')


if __name__ == '__main__':
    shaper_4f_routine()
