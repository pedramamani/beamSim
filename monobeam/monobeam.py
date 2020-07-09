import matplotlib.pyplot as plt
import numpy as np
from monobeam_config import *
from gc import collect
from pyfftw import FFTW, export_wisdom, import_wisdom
from os import path
from pickle import load, dump


class MonoBeam:
    def __init__(self, ν, Δx, I0=None, Dx=None, Nx=None):
        """
        Specify a coherent monochromatic input beam with a Gaussian cross-sectional profile.
        :param ν: frequency of beam
        :param Δx: FWHM of the cross-sectional intensity
        :param I0: beam intensity at its center
        :param Dx: width of the sampling region
        :param Nx: number of points to sample
        """
        I0 = I0 or DEFAULT_I0
        Dx = Dx or fx * Δx
        self.Nx = Nx or DEFAULT_Nx
        self.dx = Dx / self.Nx
        self.xs = np.roll(np.arange(self.Nx) - self.Nx // 2, self.Nx // 2) * self.dx

        self.λ = c / ν
        self.Rp = np.inf
        self.z = 0
        self.z0 = 0
        self.w0 = Δx / np.sqrt(2 * np.log(2))
        E0 = np.sqrt(2 * μ0 * I0)
        self.E = np.asarray(E0 * np.exp(-self.xs ** 2 / self.w0 ** 2), dtype=np.complex128)

    def rotate(self, α):
        """
        Rotate the beam by applying a linear phase in position.
        :param α: angle to rotate by
        :raises RuntimeError: if phase aliasing occurs upon rotation
        :return: MonoBeam object for chaining
        """
        if self.dx >= (self.λ / (2 * np.sin(np.abs(α))) if α != 0 else np.inf):
            raise RuntimeError(ERROR.ROTATION.format(
                int(np.ceil(np.log2(self.Nx * self.dx * 2 * np.sin(np.abs(α)) / self.λ)))))
        self._add_phase(np.tan(α) * self.xs)
        return self

    def mask(self, M):
        """
        Apply a complex mask to modulate the beam amplitude and phase.
        :param M: mask function that maps a given position to its complex multiplier
        :return: MonoBeam object for chaining
        """
        self.E *= np.asarray([M(x) for x in self.xs])
        return self

    def propagate(self, Δz):
        """
        Propagate the beam in free space.
        :param Δz: distance to propagate by
        :return: MonoBeam object for chaining
        """
        zR = π * self.w0 ** 2 / self.λ
        dw = self.z - self.z0
        propagator = (np.isinf(self.Rp), np.abs(dw + Δz) < fR * zR)

        if propagator == PROPAGATOR.IN2IN:
            self._propagate_p2p(Δz=Δz)
        elif propagator == PROPAGATOR.IN2OUT:
            self._propagate_p2p(Δz=-dw)
            self._propagate_w2s(Δz=Δz + dw)
            self.Rp = Δz + dw
        elif propagator == PROPAGATOR.OUT2IN:
            self._propagate_s2w(Δz=-dw)
            self._propagate_p2p(Δz=Δz + dw)
            self.Rp = np.inf
        else:
            self._propagate_s2w(Δz=-dw)
            self._propagate_w2s(Δz=Δz + dw)
            self.Rp = Δz + dw
        collect()
        return self

    def lens(self, f):
        """
        Simulate the effect of a thin lens on the beam by applying a quadratic phase in position.
        :param f: focal length of the lens (positive/negative for a convex/concave lens)
        :return: MonoBeam object for chaining
        """
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
        propagator = (np.isinf(self.Rp), np.abs(dw) < fR * zR)
        Rp = np.inf if propagator[1] else dw

        if propagator == PROPAGATOR.IN2IN:
            a = 1 / f
        elif propagator == PROPAGATOR.IN2OUT:
            a = 1 / f + 1 / Rp
        elif propagator == PROPAGATOR.OUT2IN:
            a = 1 / f - 1 / self.Rp
        else:
            a = 1 / f - 1 / self.Rp + 1 / Rp

        self._add_phase(Δz=-np.abs(self.xs) ** 2 * a / 2)
        self.z0 = self.z - dw
        self.w0 = w0
        self.Rp = Rp
        collect()
        return self

    def phase_section(self):
        """
        :return: Beam phase profile (cross-section plane may be spherical)
        """
        ϕs = np.roll(np.arctan2(self.E.imag, self.E.real), self.Nx // 2)
        ϕsu = np.unwrap(ϕs)
        ϕsu += ϕs[self.Nx // 2] - ϕsu[self.Nx // 2]  # unwrapping should leave center phase unchanged
        return ϕsu

    def amplitude_section(self):
        """
        :return: Beam amplitude profile (cross-section plane may be spherical)
        """
        return np.roll(np.abs(self.E), self.Nx // 2)

    def intensity_section(self):
        """
        :return: Beam intensity profile (cross-section plane may be spherical)
        """
        return np.roll(np.abs(self.E) ** 2 / (2 * μ0), self.Nx // 2)

    def plot_phase_section(self, title=PLOT.PHASE_TITLE):
        """
        Plot beam phase profile (cross-section plane may be spherical)
        :param title: title of the plot
        :return: MonoBeam object for chaining
        """
        self._plot_section(ys=self.phase_section(), title=title, label=PLOT.PHASE_LABEL)
        return self

    def plot_amplitude_section(self, title=PLOT.AMPLITUDE_TITLE):
        """
        Plot beam amplitude profile (cross-section plane may be spherical)
        :param title: title of the plot
        :return: MonoBeam object for chaining
        """
        self._plot_section(ys=self.amplitude_section() * PLOT.AMPLITUDE_SCALE, title=title, label=PLOT.AMPLITUDE_LABEL)
        return self

    def plot_intensity_section(self, title=PLOT.INTENSITY_TITLE):
        """
        Plot beam intensity profile (cross-section plane may be spherical)
        :param title: title of the plot
        :return: MonoBeam object for chaining
        """
        self._plot_section(ys=self.intensity_section() * PLOT.INTENSITY_SCALE, title=title, label=PLOT.INTENSITY_LABEL)
        return self

    def _add_phase(self, Δz):
        self.E *= np.where(self.E != 0, np.exp(-2 * π * i * Δz / self.λ), 0)

    def _propagate_p2p(self, Δz):
        if np.abs(Δz) < δz:
            return

        self._compute_fftw(direction='FFTW_FORWARD')
        inverse_squared = np.roll(((np.arange(self.Nx) - self.Nx / 2) / (self.Nx * self.dx)) ** 2, self.Nx // 2)
        self.E *= np.exp((complex(0, 1) * π * self.λ * Δz) * inverse_squared)
        self._compute_fftw(direction='FFTW_BACKWARD')
        # self._add_phase(Δz=Δz)
        self.z += Δz

    def _propagate_w2s(self, Δz):
        self._add_phase(Δz=self.xs ** 2 / (2 * Δz))
        if Δz >= 0:
            self._compute_fftw(direction='FFTW_FORWARD')
        else:
            self._compute_fftw(direction='FFTW_BACKWARD')

        self._add_phase(Δz=Δz)
        dx = self.λ * np.abs(Δz) / (self.dx * self.Nx)
        self.xs *= dx / self.dx
        self.dx = dx
        self.z += Δz

    def _propagate_s2w(self, Δz):
        dx = self.λ * np.abs(Δz) / (self.dx * self.Nx)
        self.xs *= dx / self.dx
        self.dx = dx
        if Δz >= 0:
            self._compute_fftw(direction='FFTW_FORWARD')
        else:
            self._compute_fftw(direction='FFTW_BACKWARD')

        self._add_phase(Δz=self.xs ** 2 / (2 * Δz) + Δz)
        self.z += Δz

    def _compute_fftw(self, direction):
        wisdom_path = ASSETS_DIR / WISDOM_FILE.format(self.Nx)
        if path.exists(wisdom_path):
            with open(wisdom_path, 'rb') as file:
                import_wisdom(load(file))

        fftw_object = FFTW(self.E, self.E, direction=direction, flags=['FFTW_UNALIGNED', 'FFTW_ESTIMATE'])
        self.E = fftw_object() * (1 / np.sqrt(self.Nx) if direction == 'FFTW_FORWARD' else np.sqrt(self.Nx))

        if not path.exists(wisdom_path):
            with open(wisdom_path, 'wb') as file:
                dump(export_wisdom(), file, 2)

    @staticmethod
    def _nudge(x, y):
        if np.abs(x - y) < δz:
            return y
        return x

    def _plot_section(self, ys, title, label):
        if np.isinf(self.Rp):
            print(PLOT.LOG_PLANAR.format(title))
        else:
            print(PLOT.LOG_SPHERICAL.format(title, self.Rp * 1E2))

        plt.plot(np.roll(self.xs, self.Nx // 2) * PLOT.POSITION_SCALE, ys)
        plt.title(title)
        plt.xlabel(xlabel=PLOT.POSITION_LABEL)
        plt.ylabel(ylabel=label)
        plt.show()
