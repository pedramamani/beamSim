from polybeam import PolyBeam
import numpy as np


def routine_4f_shaper(M):
    # beam = PolyBeam(f0=375, Δf=30, Δx=3, ηf=1.2, Nf=2 ** 14, ηx=2, Nx=2 ** 10)
    beam = PolyBeam(f0=375, Δf=10, Δx=2, ηf=6.25, Nf=2 ** 8, ηx=2, Nx=2 ** 12)

    # beam.rotate(α=3)
    # beam.chirp(α=0.01)
    beam.disperse(d=1500, α=45)
    beam.propagate(Δz=30).lens(f=30).propagate(Δz=30)
    beam.mask(M=M)
    beam.propagate(Δz=30).lens(f=30).propagate(Δz=30)
    beam.disperse(d=1500, α=45)  # negative dispersion to collimate beam

    beam.plot_intensity_frequency(cmap=False)  # .plot_phase_frequency()
    # beam.plot_intensity_time()  # .plot_phase_time()
    beam.plot_wigner()


if __name__ == '__main__':
    routine_4f_shaper(lambda xs: np.where(np.abs(xs) < 0.05 * 16, 0, 1))  # 4 fiber notch
    # routine_4f_shaper(lambda xs: np.where(np.abs(xs) < 2, 1, 0))  # sharp slit
