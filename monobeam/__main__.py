from monobeam import MonoBeam


def shaper_4f_routine():
    beam = MonoBeam(ν=375, Δx=3, Nx=2 ** 16)
    beam.rotate(α=3)
    beam.plot_intensity_section(title='Input Intensity').plot_phase_section(title='Input Phase')

    beam.propagate(Δz=30).lens(f=30).propagate(Δz=60).lens(f=30).propagate(Δz=30)
    beam.rotate(α=3)

    beam.plot_intensity_section(title='Output Intensity').plot_phase_section(title='Output Phase')


if __name__ == '__main__':
    shaper_4f_routine()
