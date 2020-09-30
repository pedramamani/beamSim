from monobeam import MonoBeam


def shaper_4f_routine():
    beam = MonoBeam(f=375, Δx=3, ηx=10, Nx=2 ** 12)
    beam.rotate(α=3)
    beam.plot_intensity(title='Input Intensity').plot_phase(title='Input Phase')

    beam.propagate(Δz=30).lens(f=30).propagate(distance=60).lens(f=30).propagate(distance=30)
    beam.rotate(α=3)

    beam.plot_intensity(title='Output Intensity').plot_phase(title='Output Phase', filter_=True)


if __name__ == '__main__':
    shaper_4f_routine()
