from monoBeam import MonoBeam
from polyBeam import PolyBeam
import numpy as np


def mono4fShaper():
    beam = MonoBeam(f=375, Δx=3, ηx=10, Nx=2 ** 12)
    beam.rotate(α=3).propagate(Δz=30).lens(f=30).propagate(Δz=30)
    beam.propagate(Δz=30).lens(f=30).propagate(Δz=30).rotate(α=3)
    beam.plotIntensity().plotPhase(filter_=True)


def poly4fShaper():
    beam = PolyBeam(f0=375, Δf=10, Δx=2, ηf=6.25, Nf=2 ** 8, ηx=2, Nx=2 ** 12)
    beam.rotate(α=3).disperse(d=1500, α=45).propagate(Δz=30).lens(f=30).propagate(Δz=30)
    beam.mask(M=lambda xs: np.where(np.abs(xs + 15) < 0.2, 0, 1))
    beam.propagate(Δz=30).lens(f=30).propagate(Δz=30).disperse(d=1500, α=45).rotate(α=3)
    beam.plotIntensityFrequency().plotPhaseFrequency(filter_=True)
    beam.plotWigner()


def monoDoubleSlit():
    # two slits from -0.25mm to -0.2mm and 0.2mm to 0.25mm
    def mask(xs): return np.where(
        np.logical_or(
            np.logical_and(-0.25 < xs, xs < -0.2),
            np.logical_and(0.2 < xs, xs < 0.25)
        ), 1, 0)

    beam = MonoBeam(f=400, Δx=2, I0=1, ηx=2 ** 4, Nx=2 ** 14)
    beam.mask(M=mask)
    beam.propagate(Δz=20)
    beam.plotIntensity()


def polyDoubleSlit():

    # two slits from -0.25mm to -0.2mm and 0.2mm to 0.25mm
    def mask(xs): return np.where(
        np.logical_or(
            np.logical_and(-0.25 < xs, xs < -0.2),
            np.logical_and(0.2 < xs, xs < 0.25)
        ), 1, 0)

    beam = PolyBeam(f0=400, I0=1, Δf=40, ηf=2 ** 4, Nf=2 ** 8,
                    Δx=2, ηx=2 ** 2, Nx=2 ** 13)
    beam.mask(M=mask)
    beam.propagate(Δz=20)
    beam.plotIntensityTime()


if __name__ == '__main__':
    # mono4fShaper()
    # poly4fShaper()
    # monoDoubleSlit()
    polyDoubleSlit()
