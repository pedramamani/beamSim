import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette
from monobeam import MonoBeam
from polybeam_config import *
from common import pyfftw_wisdom
import pyfftw
import multiprocessing
import time
import gc
import warnings


def log_and_cache(func):  # decorate class methods to log their execution and cache their output
    def decorate(self, *args, **kwargs):
        method = eval(f'WRAP.{func.__name__}')
        if method.type == WRAP.Type.getter:
            index = len(self.history) - 1
            while index >= 0:
                method_hist = self.history[index]
                if method_hist.type:
                    break
                if method_hist is method:
                    return getattr(self, method.attribute)
                index -= 1

        start_time = time.time()
        func(self, *args, **kwargs)
        if VERBOSE and method.type != WRAP.Type.getter:
            print(method.message.format(**kwargs), WRAP.time.format(time.time() - start_time))
        self.history.append(method)
        gc.collect()

        if method.type == WRAP.Type.modifier:
            return self
        elif method.type == WRAP.Type.getter:
            return getattr(self, method.attribute)

    return decorate


def plot(func):  # decorate plotter methods to automagically plot
    _, variable_name, versus_name = func.__name__.split('_')
    method = f'get_{variable_name}_{versus_name}'
    variable = eval(f'PLOT.{variable_name}')
    versus = eval(f'PLOT.{versus_name}')

    def decorate(self, title=variable.title, cmap=PLOT_CMAP, **kwargs):
        values = np.real(getattr(self, method)(**kwargs))
        if cmap:
            vs = values * variable.scale
            self._plot_cmap(vs, title, variable.label, versus)
        else:
            vs = values[self.Nx // 2] * variable.scale
            self._plot_line(vs, title, variable.label, versus)
        return self

    return decorate


class PolyBeam:
    @log_and_cache
    def __init__(self, f0, Δf, Δx, ηf=None, Nf=None, ηx=None, Nx=None, I0=None):
        """
        Specify a polychromatic input beam with a Gaussian cross-sectional profile and spectral envelope.
        :param f0: central frequency of the beam (THz)
        :param Δf: FWHM of the spectral intensity (THz)
        :param Δx: FWHM of the cross-sectional intensity (mm)
        :param ηf: ratio of sampling region to beam FWHM in frequency
        :param Nf: number of points to sample in frequency
        :param ηx: ratio of sampling region to beam FWHM in position
        :param Nx: number of points to sample in position
        :param I0: beam intensity at its center (TW/m²)
        """
        pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

        self.f0 = f0 * PRE.T
        self.Δf = Δf * PRE.T
        self.Δx = Δx * PRE.m
        self.Df = (DEFAULT_ηf if ηf is None else ηf) * self.Δf
        self.Nf = DEFAULT_Nf if Nf is None else Nf
        self.Dx = (DEFAULT_ηx if ηx is None else ηx) * self.Δx
        self.Nx = DEFAULT_Nx if Nx is None else Nx
        I0 = DEFAULT_I0 if I0 is None else I0  # (TW/m²)

        self.df = self.Df / self.Nf
        self.fs = np.arange(start=self.f0 - self.Df / 2, stop=self.f0 + self.Df / 2, step=self.df)
        self.ts = np.arange(start=-1 / (2 * self.df), stop=1 / (2 * self.df), step=1 / self.Df)

        self.beams = []
        for f in self.fs:
            I = I0 * np.exp(-4 * np.log(2) * ((self.f0 - f) / self.Δf) ** 2)
            self.beams.append(MonoBeam(f=f / PRE.T, Δx=Δx, I0=I, ηx=self.Dx / self.Δx, Nx=self.Nx))

        self.Et = None
        self.φt = None
        self.At = None
        self.It = None
        self.Ef = None
        self.φf = None
        self.Af = None
        self.If = None
        self.W = None
        self.history = []

    @log_and_cache
    def rotate(self, α):
        """
        Rotate the beam by applying a linear phase in position.
        :param α: angle to rotate by (°)
        :return: PolyBeam object for chaining
        """
        for beam in self.beams:
            beam.rotate(α=α)

    @log_and_cache
    def mask(self, M):  # todo: fix mask and rotate for non-planar
        """
        Apply a complex mask to modulate the beam amplitude and phase.
        :param M: mask function that maps list of position (mm) to their complex multiplier
        :return: PolyBeam object for chaining
        """
        for beam in self.beams:
            beam.mask(M=M)

    @log_and_cache
    def propagate(self, Δz):
        """
        Propagate the beam in free space.
        :param Δz: distance to propagate by (cm)
        :return: PolyBeam object for chaining
        """
        for beam in self.beams:
            beam.propagate(Δz=Δz)
        self.Dx = self.beams[self.Nf // 2].Dx

    @log_and_cache
    def lens(self, f):
        """
        Simulate the effect of a thin lens on the beam by applying a quadratic phase in position.
        :param f: focal length of the lens (cm)
        :return: PolyBeam object for chaining
        """
        for beam in self.beams:
            beam.lens(f=f)

    @log_and_cache
    def disperse(self, d, α):
        """
        Simulate a dispersive first-order diffraction by applying frequency-dependent rotation.
        :param d: grating groove density (1/mm)
        :param α: incident beam angle (°)
        :raises RuntimeError: if first order diffraction does not exist for some frequencies
        :raises RuntimeError: if phase aliasing occurs upon dispersion
        :return: PolyBeam object for chaining
        """
        δ = PRE.m / d
        α = np.deg2rad(α)
        if np.isnan(np.arcsin(c / (self.fs[0] * δ) - np.sin(α)) - np.arcsin(c / (self.fs[-1] * δ) - np.sin(α))):
            raise RuntimeError(ERROR.disperse)
        θ0 = np.arcsin(c / (self.f0 * δ) - np.sin(α))
        for beam, f in zip(self.beams, self.fs):
            beam.rotate(α=np.rad2deg(np.arcsin(c / (f * δ) - np.sin(α)) - θ0))

    @log_and_cache
    def chirp(self, α):
        """
        Apply a quadratic phase shift in frequency to chirp the beam.
        :param α: the chirp rate (ps²)
        :return: PolyBeam object for chaining
        """
        α *= PRE.p ** 2
        if self.Nf < 4 * π * α * self.Df ** 2:  # require: 4πα Df^2 < Nf
            raise RuntimeError(ERROR.chirp_alias.format(
                self.Nf / (4 * π * self.Df ** 2) / PRE.f ** 2,
                np.ceil(np.log2(4 * π * α * self.Df ** 2)),
                np.sqrt(self.Nf / (4 * π * α)) / self.Δf))

        for f, beam in zip(self.fs, self.beams):
            m = np.exp(4 * π ** 2 * i * α * (f - self.f0) ** 2)
            beam.mask(M=lambda xs: m)

    @log_and_cache
    def get_field_time(self):
        """
        Fourier transform beam from frequency to time domain and center it.
        :return: beam complex field values in time across its cross-section after removing the carrier frequency
        :raises RuntimeError: if any of the component beams is not planar
        """
        if any([np.isfinite(b.Rp) for b in self.beams]):
            raise RuntimeError(ERROR.transform_nonplanar)

        self.Et = np.roll(self.get_field_frequency(), self.Nf // 2, axis=1)
        with pyfftw_wisdom(ASSETS_DIR / WISDOM_FILE.format(self.Nf)):
            for j, E in enumerate(self.Et):
                fftw = pyfftw.FFTW(E, E, direction='FFTW_BACKWARD', flags=['FFTW_UNALIGNED', 'FFTW_ESTIMATE'])
                self.Et[j] = fftw() * np.sqrt(self.Nf)
        self.Et = np.roll(self.Et, self.Nf // 2, axis=1)

    @log_and_cache
    def get_phase_time(self, filter_=FILTER_PHASE):
        """
        :param filter_: whether to filter phase values with low amplitude
        :return: beam temporal phase profile across its cross-section after removing the carrier frequency
        """
        Et = self.get_field_time()
        φt = np.arctan2(Et.imag, Et.real)
        self.φt = np.unwrap(φt, axis=1)
        for j in np.arange(self.Nx):
            self.φt[j] += φt[j][self.Nf // 2] - self.φt[j][self.Nf // 2]  # unwrapping should not change center phase
        if filter_:
            self.φt = np.where(np.abs(Et) > np.amax(np.abs(Et)) * ε, self.φt, np.nan)

    @log_and_cache
    def get_amplitude_time(self):
        """
        :return: beam temporal amplitude profile across its cross-section
        """
        self.At = np.abs(self.get_field_time())

    @log_and_cache
    def get_intensity_time(self):
        """
        :return: beam temporal intensity profile across its cross-section
        """
        self.It = np.abs(self.get_field_time()) ** 2 / (2 * μ0)

    @log_and_cache
    def get_field_frequency(self):
        """
        :return: beam complex field values in frequency across its cross-section
        """
        self.Ef = np.transpose([b.get_field() for b in self.beams])

    @log_and_cache
    def get_phase_frequency(self, filter_=FILTER_PHASE):
        """
        :param filter_: whether to filter values with low amplitude
        :return: beam spectral phase profile across its cross-section
        """
        φf = np.transpose([b.get_phase() for b in self.beams])
        self.φf = np.unwrap(φf, axis=1)
        for j in np.arange(self.Nx):
            self.φf[j] += φf[j][self.Nf // 2] - self.φf[j][self.Nf // 2]  # unwrapping should not change center phase
        if filter_:
            Af = self.get_amplitude_frequency()
            self.φf = np.where(Af > np.amax(Af) * ε, self.φf, np.nan)

    @log_and_cache
    def get_amplitude_frequency(self):
        """
        :return: beam spectral amplitude profile across its cross-section
        """
        self.Af = np.transpose([b.get_amplitude() for b in self.beams])

    @log_and_cache
    def get_intensity_frequency(self):
        """
        :return: beam spectral intensity profile across its cross-section
        """
        self.If = np.transpose([b.get_intensity() for b in self.beams])

    @log_and_cache
    def get_wigner(self):
        """
        :return: Wigner distribution of the central beam at position x=0
        """
        E = self.get_field_time()[self.Nx // 2]
        Et = np.pad(E, (0, self.Nf))
        Es = np.pad(np.flip(np.conj(E)), (0, self.Nf))

        plt.plot(np.pad(np.real(E), (self.Nf, self.Nf)))
        plt.show()

        self.W = np.ndarray(shape=(self.Nf, self.Nf), dtype=np.complex128)

        for t in range(self.Nf):
            self.W[t] = np.multiply(Et[t: t + self.Nf], Es[self.Nf - t - 1: 2 * self.Nf - t - 1]) / (2 * μ0)
            # self.W[t] = np.multiply(E[t: t + self.Nf], E[self.Nf - t - 1: 2 * self.Nf - t - 1]) / (2 * μ0)

        with pyfftw_wisdom(ASSETS_DIR / WISDOM_FILE.format(self.Nf)):
            for t, Wt in enumerate(self.W):
                fftw = pyfftw.FFTW(Wt, Wt, direction='FFTW_FORWARD', flags=['FFTW_UNALIGNED', 'FFTW_ESTIMATE'])
                self.W[t] = fftw() / np.sqrt(self.Nf)
        self.W = np.roll(self.W.T, self.Nf // 2, axis=0)

    @plot
    def plot_field_time(self, title=None, cmap=None):
        """
        Plot beam real field values in time after removing the carrier frequency.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central field values
        :return: PolyBeam object for chaining
        """

    @plot
    def plot_phase_time(self, title=None, cmap=None, filter_=None):
        """
        Plot beam temporal phase profile after subtracting entral carrier frequency.
        :param title: title of the plot
        :param filter_: whether to filter values with low amplitude
        :param cmap: whether to plot a color map across beam cross-section or only the central phase values
        :return: PolyBeam object for chaining
        """

    @plot
    def plot_amplitude_time(self, title=None, cmap=None):
        """
        Plot beam temporal amplitude profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central amplitude values
        :return: PolyBeam object for chaining
        """

    @plot
    def plot_intensity_time(self, title=None, cmap=None):
        """
        Plot beam temporal intensity profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central intensity values
        :return: PolyBeam object for chaining
        """

    @plot
    def plot_field_frequency(self, title=None, cmap=None):
        """
        Plot beam real field values in frequency.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central field values
        :return: PolyBeam object for chaining
        """

    @plot
    def plot_phase_frequency(self, title=None, cmap=None, filter_=None):
        """
        Plot beam spectral phase profile.
        :param title: title of the plot
        :param filter_: whether to filter values with low amplitude
        :param cmap: whether to plot a color map across beam cross-section or only the central phase values
        :return: PolyBeam object for chaining
        """

    @plot
    def plot_amplitude_frequency(self, title=None, cmap=None):
        """
        Plot beam spectral amplitude profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central amplitude values
        :return: PolyBeam object for chaining
        """

    @plot
    def plot_intensity_frequency(self, title=None, cmap=None):
        """
        Plot beam spectral intensity profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central intensity values
        :return: PolyBeam object for chaining
        """

    def plot_wigner(self, title=PLOT.wigner.title):
        """
        Plot the Wigner distribution of the central beam at position x=0.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        with color_palette('husl'):
            Rt = (self.ts[0] * PLOT.time.scale, self.ts[-1] * PLOT.time.scale)
            Rf = (self.fs[0] * PLOT.frequency.scale, self.fs[-1] * PLOT.frequency.scale)
            W = self.get_wigner() * PLOT.intensity.scale
            It = np.abs(np.sum(W, axis=0))
            If = np.abs(np.sum(W, axis=1))

            fig = plt.figure()
            area_main = fig.add_axes([0.1, 0.12, 0.86, 0.8])
            area_main.set_title(title)
            area_main.set_axis_off()

            gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
                                  left=0.12, right=0.81, bottom=0.12, top=0.9, wspace=0.02, hspace=0.02)
            area_W = fig.add_subplot(gs[1, 0])
            area_It = fig.add_subplot(gs[0, 0])
            area_If = fig.add_subplot(gs[1, 1])

            cmap = area_W.imshow(np.abs(W), aspect='auto', extent=[*Rt, *Rf], interpolation='none')
            area_W.set_xlabel(PLOT.time.label)
            area_W.set_ylabel(PLOT.frequency.label)
            plt.colorbar(cmap, ax=area_main).set_label(PLOT.intensity.label)

            area_If.plot(If, self.fs * PLOT.frequency.scale, 'b')
            area_If.margins(y=0)
            area_If.set_xticks([])
            area_If.set_yticks([])

            area_It.plot(self.ts * PLOT.time.scale, It, 'b')
            area_It.margins(x=0)
            area_It.set_xticks([])
            area_It.set_yticks([])

            plt.gcf().canvas.set_window_title('-'.join(title.lower().split()))
            with warnings.catch_warnings():  # matplotlib throws a UserWarning that I can't fix!
                warnings.simplefilter('ignore')
                plt.show()
        return self

    def _plot_line(self, values, title, label, versus):  # todo: fix non-matching x positions
        if any([np.isfinite(b.Rp) for b in self.beams]):
            raise RuntimeError(ERROR.plot_nonplanar)

        versus_values = getattr(self, versus.attribute) * versus.scale
        plt.plot(versus_values, values)
        plt.xlim(versus_values[0], versus_values[-1])
        plt.title(title)
        plt.gcf().canvas.set_window_title('-'.join(title.lower().split() + [versus.title.lower()]))
        plt.xlabel(versus.label)
        plt.ylabel(label)
        plt.show()

    def _plot_cmap(self, values, title, label, versus):
        with color_palette('husl'):
            if any([np.isfinite(b.Rp) for b in self.beams]):
                raise RuntimeError(ERROR.plot_nonplanar)

            versus_values = getattr(self, versus.attribute) * versus.scale
            wx = self.Dx / 2 * PLOT.position.scale

            plt.imshow(values, aspect='auto', extent=[versus_values[0], versus_values[-1], -wx, wx],
                       interpolation='none')
            plt.title(title)
            plt.gcf().canvas.set_window_title('-'.join(title.lower().split() + [versus.title.lower()]))
            plt.xlabel(versus.label)
            plt.ylabel(PLOT.position.label)
            plt.colorbar().set_label(label)
            plt.show()
