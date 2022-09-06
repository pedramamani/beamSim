import numpy as np
import matplotlib.pyplot as plt
import pyfftw
import multiprocessing
from config import *


DEFAULT_ηx = 16
DEFAULT_Nx = 2 ** 16
DEFAULT_I0 = 1  # (TW/m²)


class MonoBeam:
    def __init__(self, f, Δx, I0=None, ηx=None, Nx=None):
        """
        Specify a coherent monochromatic input beam with a Gaussian cross-sectional profile.
        :param f: frequency of beam (THz)
        :param Δx: FWHM of the cross-sectional intensity (mm)
        :param I0: beam intensity at its center (TW/m²)
        :param ηx: ratio of sampling region to beam FWHM
        :param Nx: number of points to sample
        """
        pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

        self.f = f * PRE.T
        self.Δx = Δx * PRE.m
        self.I0 = (DEFAULT_I0 if I0 is None else I0) * PRE.T
        self.Dx = (DEFAULT_ηx if ηx is None else ηx) * self.Δx
        self.Nx = DEFAULT_Nx if Nx is None else Nx

        self.dx = self.Dx / self.Nx
        self.xs = np.roll(np.arange(start=-self.Dx / 2, stop=self.Dx / 2, step=self.dx), self.Nx // 2)
        self.λ = c / self.f
        self.Rp = np.inf
        self.z = 0
        self.z0 = 0
        self.w0 = self.Δx / np.sqrt(2 * np.log(2))
        E0 = np.sqrt(2 * μ0 * self.I0)
        self.E = np.asarray(E0 * np.exp(-self.xs ** 2 / self.w0 ** 2), dtype=np.complex128)

    def rotate(self, α):
        """
        Rotate the beam by applying a linear phase in position.
        :param α: angle to rotate by (°)
        :raises RuntimeError: if phase aliasing occurs upon rotation
        :return: MonoBeam object for chaining
        """
        α = np.deg2rad(α)
        if self.dx >= (self.λ / (2 * np.sin(np.abs(α))) if α != 0 else np.inf):  # require: 2dx sin(|α|) < λ
            raise RuntimeError(ERROR.ROTATE_ALIAS.format(
                np.rad2deg(np.arcsin(self.λ / (2 * self.dx))),
                np.ceil(np.log2(self.Dx * 2 * np.sin(np.abs(α)) / self.λ)),
                self.λ * self.Nx / (2 * np.sin(np.abs(α)) * self.Δx)))
        self._addPhase(np.tan(α) * self.xs)
        return self

    def mask(self, M):
        """
        Apply a complex mask to modulate the beam amplitude and phase.
        :param M: mask function that maps a given position to its complex multiplier (mm → ℂ)
        :return: MonoBeam object for chaining
        """
        M0 = np.complex(M(0))
        φ0 = np.arctan2(M0.imag, M0.real)  # shift phase values to keep center phase unchanged
        self.E = np.multiply(self.E, M(self.xs / PRE.m) * np.exp(-φ0))
        return self

    def propagate(self, Δz):
        """
        Propagate the beam in free space.
        :param Δz: distance to propagate by (cm)
        :return: MonoBeam object for chaining
        """
        Δz *= PRE.c
        zR = π * self.w0 ** 2 / self.λ
        dw = self.z - self.z0
        propagator = (np.isinf(self.Rp), np.abs(dw + Δz) < ηR * zR)

        if propagator == PROPAGATOR.IN_TO_IN:
            self._propagateP2P(Δz=Δz)
        elif propagator == PROPAGATOR.IN_TO_OUT:
            self._propagateP2P(Δz=-dw)
            self._propagateW2S(Δz=Δz + dw)
            self.Rp = Δz + dw
        elif propagator == PROPAGATOR.OUT_TO_IN:
            self._propagateS2W(Δz=-dw)
            self._propagateP2P(Δz=Δz + dw)
            self.Rp = np.inf
        else:
            self._propagateS2W(Δz=-dw)
            self._propagateW2S(Δz=Δz + dw)
            self.Rp = Δz + dw
        return self

    def lens(self, f):
        """
        Simulate the effect of a thin lens on the beam by applying a quadratic phase in position.
        :param f: focal length of the lens (cm)
        :return: MonoBeam object for chaining
        """
        f *= PRE.c
        dw = self._nudge(self.z - self.z0, 0)
        zR = π * self.w0 ** 2 / self.λ
        w = self.w0 * np.sqrt(1 + (dw / zR) ** 2)

        if dw != 0:  # calculate wavefront curvature radius after the lens
            R = self._nudge(dw * (1 + (zR / dw) ** 2), f)
            if R != f:
                R = 1 / (1 / R - 1 / f)
            else:
                R = np.inf
        else:  # at focus or entrance pupil input beam is planar and output beam is spherical
            R = -f

        if np.isfinite(R):  # calculate new waist radius and position
            η = self.λ * R / (π * w ** 2)
            dw = R / (1 + η ** 2)
            w0 = w / np.sqrt(1 + 1 / η ** 2)
        else:  # output beam is planar
            dw = 0
            w0 = w

        zR = π * w0 ** 2 / self.λ  # if currently inside Rayleigh distance, output beam is planar
        propagator = (np.isinf(self.Rp), np.abs(dw) < ηR * zR)
        Rp = np.inf if propagator[1] else dw

        if propagator == PROPAGATOR.IN_TO_IN:
            a = 1 / f
        elif propagator == PROPAGATOR.IN_TO_OUT:
            a = 1 / f + 1 / Rp
        elif propagator == PROPAGATOR.OUT_TO_IN:
            a = 1 / f - 1 / self.Rp
        else:
            a = 1 / f - 1 / self.Rp + 1 / Rp

        self._addPhase(Δz=-np.abs(self.xs) ** 2 * a / 2)
        self.z0 = self.z - dw
        self.w0 = w0
        self.Rp = Rp
        return self

    def getPhase(self, filter_=FILTER_PHASE):
        """
        :param filter_: whether to filter values with low amplitude
        :return: Beam phase profile (cross-section plane may be spherical)
        """
        E = self.getField()
        φ = np.arctan2(E.imag, E.real)
        φu = np.unwrap(φ)
        φu += φ[self.Nx // 2] - φu[self.Nx // 2]  # unwrapping should leave center phase unchanged
        if filter_:
            φu = np.where(np.abs(E) > np.amax(np.abs(E)) * ε, φu, np.nan)
        return φu

    def getAmplitude(self):
        """
        :return: Beam amplitude profile (cross-section plane may be spherical)
        """
        return np.roll(np.abs(self.E), self.Nx // 2)

    def getField(self):
        """
        :return: Beam field profile (cross-section plane may be spherical)
        """
        return np.roll(self.E, self.Nx // 2)

    def getIntensity(self):
        """
        :return: Beam intensity profile (cross-section plane may be spherical)
        """
        return np.roll(np.abs(self.E) ** 2 / (2 * μ0), self.Nx // 2)

    def plotPhase(self, title=PLOT.PHASE.TITLE, filter_=FILTER_PHASE):
        """
        Plot beam phase profile (cross-section plane may be spherical)
        :param title: title of the plot
        :param filter_: whether to filter values with low amplitude
        :return: MonoBeam object for chaining
        """
        values = self.getPhase(filter_=filter_) * PLOT.PHASE.SCALE
        self._plot(values, title, PLOT.PHASE.LABEL)
        return self

    def plotAmplitude(self, title=PLOT.AMPLITUDE.TITLE):
        """
        Plot beam amplitude profile (cross-section plane may be spherical)
        :param title: title of the plot
        :return: MonoBeam object for chaining
        """
        values = self.getAmplitude() * PLOT.AMPLITUDE.SCALE
        self._plot(values, title, PLOT.AMPLITUDE.LABEL)
        return self

    def plotField(self, title=PLOT.FIELD.TITLE):
        """
        Plot beam real field values (cross-section plane may be spherical)
        :return: MonoBeam object for chaining
        """
        values = self.getField() * PLOT.FIELD.SCALE
        self._plot(values, title, PLOT.FIELD.LABEL)
        return self

    def plotIntensity(self, title=PLOT.INTENSITY.TITLE):
        """
        Plot beam intensity profile (cross-section plane may be spherical)
        :param title: title of the plot
        :return: MonoBeam object for chaining
        """
        values = self.getIntensity() * PLOT.INTENSITY.SCALE
        self._plot(values, title, PLOT.INTENSITY.LABEL)
        return self

    def _addPhase(self, Δz):
        self.E *= np.where(self.E != 0, np.exp(-2 * π * i * Δz / self.λ), 0)

    def _propagateP2P(self, Δz):
        if np.abs(Δz) < δz:
            return

        self._computeFftw(direction='FFTW_FORWARD')
        inverse_squared = np.roll(((np.arange(self.Nx) - self.Nx // 2) / (self.Nx * self.dx)) ** 2, self.Nx // 2)
        self.E *= np.exp((complex(0, 1) * π * self.λ * Δz) * inverse_squared)
        self._computeFftw(direction='FFTW_BACKWARD')
        self.z += Δz

    def _propagateW2S(self, Δz):
        self._addPhase(Δz=self.xs ** 2 / (2 * Δz))
        if Δz >= 0:
            self._computeFftw(direction='FFTW_FORWARD')
        else:
            self._computeFftw(direction='FFTW_BACKWARD')

        dx = self.λ * np.abs(Δz) / (self.dx * self.Nx)
        self.xs *= dx / self.dx
        self.Dx *= dx / self.dx
        self.dx = dx
        self.z += Δz

    def _propagateS2W(self, Δz):
        dx = self.λ * np.abs(Δz) / (self.dx * self.Nx)
        self.xs *= dx / self.dx
        self.Dx *= dx / self.dx
        self.dx = dx
        if Δz >= 0:
            self._computeFftw(direction='FFTW_FORWARD')
        else:
            self._computeFftw(direction='FFTW_BACKWARD')

        self._addPhase(Δz=self.xs ** 2 / (2 * Δz))
        self.z += Δz

    def _computeFftw(self, direction):
        with pyfftwWisdom(ASSETS_DIR / WISDOM_FILE.format(self.Nx)):
            fftw = pyfftw.FFTW(self.E, self.E, direction=direction, flags=['FFTW_UNALIGNED', 'FFTW_ESTIMATE'])
            self.E = fftw() * (1 / np.sqrt(self.Nx) if direction == 'FFTW_FORWARD' else np.sqrt(self.Nx))

    @staticmethod
    def _nudge(x, y):
        if np.isclose(x, y, rtol=δz):
            return y
        return x

    def _plot(self, ys, title, label):
        if np.isinf(self.Rp):
            print(PLOT.PLANAR_WARN.format(title))
        else:
            print(PLOT.SPHERICAL_WARN.format(title, self.Rp / PRE.c))

        plt.plot(np.roll(self.xs, self.Nx // 2) * PLOT.POSITION.SCALE, ys)
        plt.gcf().canvas.set_window_title('-'.join(title.lower().split()))
        plt.title(title)
        plt.xlabel(xlabel=PLOT.POSITION.LABEL)
        plt.ylabel(ylabel=label)
        plt.show()
