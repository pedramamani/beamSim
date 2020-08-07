from polybeam import PolyBeam
import numpy as np


def routine_4f_shaper(M):
    beam = PolyBeam(f0=375, Δf=30, Δx=3, ηf=6, Nf=2 ** 10, ηx=4, Nx=2 ** 14)

    # beam.rotate(α=3)
    beam.chirp(α=5000)
    beam.disperse(d=1500, α=45)
    beam.propagate(Δz=30).lens(f=30).propagate(Δz=30)
    beam.mask(M=M)
    beam.propagate(Δz=30).lens(f=30).propagate(Δz=30)
    beam.disperse(d=1500, α=45)  # negative dispersion to collimate beam

    beam.plot_intensity_frequency().plot_phase_frequency()
    beam.plot_intensity_time().plot_phase_time()
    beam.plot_wigner()


if __name__ == '__main__':
    routine_4f_shaper(lambda xs: np.where(np.isclose(xs, 5, rtol=0.4), 0, 1))  # 8 fiber notch
    # routine_4f_shaper(lambda xs: np.where(np.abs(xs) < 2, 1, 0))  # sharp slit
