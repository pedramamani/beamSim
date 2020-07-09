from polybeam import PolyBeam
import numpy as np


def shaper_4f_routine():
    beam = PolyBeam(ν0=3.75E14, Δν=1E13, Δx=3E-3)

    beam.plot_intensity_time(title='Input Intensity').plot_phase_time(title='Input Phase')
    beam.disperse(d=1E-3 / 1300, α=np.deg2rad(70))
    beam.plot_intensity_section(title='Input Intensity').plot_phase_section(title='Input Phase')

    beam.chirp(1E-7 / 1E12)  # units for chirp?

    beam.propagate(Δz=0.3).lens(f=0.3).propagate(Δz=0.6).lens(f=0.3).propagate(Δz=0.3)
    beam.disperse(d=-1E-3 / 1300, α=-np.deg2rad(70))  # negative dispersion to collimate beam

    beam.plot_intensity_section(title='Output Intensity').plot_phase_section(title='Output Phase')
    beam.plot_intensity_time(title='Output Intensity').plot_phase_time(title='Output Phase')


if __name__ == '__main__':
    shaper_4f_routine()
