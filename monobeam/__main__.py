from monobeam import MonoBeam
import numpy as np


def shaper_4f_routine():
    beam = MonoBeam(ν=3.75E14, Δx=3E-3, Nx=2 ** 16)
    beam.rotate(α=np.deg2rad(3))
    beam.plot_intensity_section(title='Input Intensity')
    beam.plot_phase_section(title='Input Phase')

    beam.lens(f=0.3)
    beam.propagate(Δz=0.6)
    beam.lens(f=0.3)

    beam.plot_intensity_section(title='Output Intensity')
    beam.plot_phase_section(title='Output Phase')


if __name__ == '__main__':
    shaper_4f_routine()
