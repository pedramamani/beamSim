from monobeam import MonoBeam
from polybeam import PolyBeam
import numpy as np


def monobeam_4f_shaper():
    beam = MonoBeam(f=375, Δx=3, ηx=10, Nx=2 ** 12)
    beam.rotate(α=3).propagate(Δz=30).lens(f=30).propagate(Δz=30)
    beam.propagate(Δz=30).lens(f=30).propagate(Δz=30).rotate(α=3)
    beam.plot_intensity().plot_phase(filter_=True)


def polybeam_4f_shaper():
    beam = PolyBeam(f0=375, Δf=10, Δx=2, ηf=6.25, Nf=2 ** 8, ηx=2, Nx=2 ** 12)
    beam.rotate(α=3).disperse(d=1500, α=45).propagate(Δz=30).lens(f=30).propagate(Δz=30)
    beam.mask(M=lambda xs: np.where(np.abs(xs + 15) < 0.2, 0, 1))
    beam.propagate(Δz=30).lens(f=30).propagate(Δz=30).disperse(d=1500, α=45).rotate(α=3)
    beam.plot_intensity_frequency().plot_phase_frequency(filter_=True)
    beam.plot_wigner()


if __name__ == '__main__':
    # monobeam_4f_shaper()
    polybeam_4f_shaper()
