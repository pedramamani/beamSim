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


class PolyBeam:
    def __init__(self, f0, f_fwhm, x_fwhm, f_ratio=None, nf=None, x_ratio=None, nx=None, intensity=None):
        """
        Specify a polychromatic input beam with a Gaussian cross-sectional profile and spectral envelope.
        :param f0: central frequency of the beam (THz)
        :param f_fwhm: FWHM of the spectral intensity (THz)
        :param x_fwhm: FWHM of the cross-sectional intensity (mm)
        :param f_ratio: ratio of sampling region to beam FWHM in frequency
        :param nf: number of points to sample in frequency
        :param x_ratio: ratio of sampling region to beam FWHM in position
        :param nx: number of points to sample in position
        :param intensity: beam intensity at its center (TW/m²)
        """
        start_time = time.time()
        pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

        self.f0 = f0 * PRE.T
        self.f_fwhm = f_fwhm * PRE.T
        self.x_fwhm = x_fwhm * PRE.m
        self.Df = (DEFAULT_ηf if f_ratio is None else f_ratio) * self.f_fwhm
        self.Nf = DEFAULT_Nf if nf is None else nf
        self.Dx = (DEFAULT_ηx if x_ratio is None else x_ratio) * self.x_fwhm
        self.Nx = DEFAULT_Nx if nx is None else nx
        intensity = DEFAULT_I0 if intensity is None else intensity  # (TW/m²)

        self.df = self.Df / self.Nf
        self.fs = np.arange(start=self.f0 - self.Df / 2, stop=self.f0 + self.Df / 2, step=self.df)
        self.ts = np.arange(start=-1 / (2 * self.df), stop=1 / (2 * self.df), step=1 / self.Df)

        self.beams = []
        for f in self.fs:
            I = intensity * np.exp(-4 * np.log(2) * ((self.f0 - f) / self.f_fwhm) ** 2)
            self.beams.append(MonoBeam(f=f / PRE.T, Δx=x_fwhm, I0=I, ηx=self.Dx / self.x_fwhm, Nx=self.Nx))

        self.Et = None
        self.phase_t = None
        self.At = None
        self.It = None
        self.Ef = None
        self.phase_f = None
        self.Af = None
        self.If = None
        self.W = None
        self.history = []
        
        if VERBOSE:
            print(LOG.INIT.format(f0, f_fwhm, x_fwhm), LOG.TIME.format(time.time() - start_time))
        self.history.append('init')
        gc.collect()

    def rotate(self, angle):
        """
        Rotate the beam by applying a linear phase in position.
        :param angle: angle to rotate by (°)
        :return: PolyBeam object for chaining
        """
        start_time = time.time()
        
        for beam in self.beams:
            beam.rotate(α=angle)
        
        if VERBOSE:
            print(LOG.ROTATE.format(angle), LOG.TIME.format(time.time() - start_time))
        self.history.append('rotate')
        gc.collect()
        return self

    def mask(self, mask):  # todo: fix mask and rotate for non-planar
        """
        Apply a complex mask to modulate the beam amplitude and phase.
        :param mask: mask function that maps list of position (mm) to their complex multiplier
        :return: PolyBeam object for chaining
        """
        start_time = time.time()

        for beam in self.beams:
            beam.mask(M=mask)

        if VERBOSE:
            print(LOG.MASK, LOG.TIME.format(time.time() - start_time))
        self.history.append('mask')
        gc.collect()
        return self

    def propagate(self, distance):
        """
        Propagate the beam in free space.
        :param distance: distance to propagate by (cm)
        :return: PolyBeam object for chaining
        """
        start_time = time.time()
        
        for beam in self.beams:
            beam.propagate(Δz=distance)
        self.Dx = self.beams[self.Nf // 2].Dx
        
        if VERBOSE:
            print(LOG.PROPAGATE.format(distance), LOG.TIME.format(time.time() - start_time))
        self.history.append('propagate')
        gc.collect()
        return self

    def lens(self, f):
        """
        Simulate the effect of a thin lens on the beam by applying a quadratic phase in position.
        :param f: focal length of the lens (cm)
        :return: PolyBeam object for chaining
        """
        start_time = time.time()

        for beam in self.beams:
            beam.lens(f=f)

        if VERBOSE:
            print(LOG.LENS.format(f), LOG.TIME.format(time.time() - start_time))
        self.history.append('lens')
        gc.collect()
        return self

    def disperse(self, d, angle):
        """
        Simulate a dispersive first-order diffraction by applying frequency-dependent rotation.
        :param d: grating groove density (1/mm)
        :param angle: incident beam angle (°)
        :raises RuntimeError: if first order diffraction does not exist for some frequencies
        :raises RuntimeError: if phase aliasing occurs upon dispersion
        :return: PolyBeam object for chaining
        """
        start_time = time.time()

        δ = PRE.m / d
        angle = np.deg2rad(angle)
        if np.isnan(np.arcsin(c / (self.fs[0] * δ) - np.sin(angle)) - np.arcsin(c / (self.fs[-1] * δ) - np.sin(angle))):
            raise RuntimeError(ERROR.DISPERSE)
        θ0 = np.arcsin(c / (self.f0 * δ) - np.sin(angle))
        for beam, f in zip(self.beams, self.fs):
            beam.rotate(α=np.rad2deg(np.arcsin(c / (f * δ) - np.sin(angle)) - θ0))

        if VERBOSE:
            print(LOG.DISPRESE.format(d, angle), LOG.TIME.format(time.time() - start_time))
        self.history.append('disperse')
        gc.collect()
        return self

    def chirp(self, rate):
        """
        Apply a quadratic phase shift in frequency to chirp the beam.
        :param rate: the chirp rate (ps²)
        :return: PolyBeam object for chaining
        """
        start_time = time.time()

        rate *= PRE.p ** 2
        if self.Nf < 4 * π * rate * self.Df ** 2:  # require: 4πα Df^2 < Nf
            raise RuntimeError(ERROR.CHIRP_ALIAS.format(
                self.Nf / (4 * π * self.Df ** 2) / PRE.f ** 2,
                np.ceil(np.log2(4 * π * rate * self.Df ** 2)),
                np.sqrt(self.Nf / (4 * π * rate)) / self.f_fwhm))

        for f, beam in zip(self.fs, self.beams):
            m = np.exp(4 * π ** 2 * i * rate * (f - self.f0) ** 2)
            beam.mask(M=lambda xs: m)

        if VERBOSE:
            print(LOG.CHIRP.format(rate), LOG.TIME.format(time.time() - start_time))
        self.history.append('chirp')
        gc.collect()
        return self

    def get_field_time(self):
        """
        Fourier transform beam from frequency to time domain and center it.
        :return: beam complex field values in time across its cross-section after removing the carrier frequency
        :raises RuntimeError: if any of the component beams is not planar
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'get_field_time':
                return self.Et
            elif method in LOG.MODIFIERS:
                break
            index -= 1
        
        if any([np.isfinite(b.Rp) for b in self.beams]):
            raise RuntimeError(ERROR.TRANSFORM_NONPLANAR)

        self.Et = np.roll(self.get_field_frequency(), self.Nf // 2, axis=1)
        with pyfftw_wisdom(ASSETS_DIR / WISDOM_FILE.format(self.Nf)):
            for j, E in enumerate(self.Et):
                fftw = pyfftw.FFTW(E, E, direction='FFTW_BACKWARD', flags=['FFTW_UNALIGNED', 'FFTW_ESTIMATE'])
                self.Et[j] = fftw() * np.sqrt(self.Nf)
        self.Et = np.roll(self.Et, self.Nf // 2, axis=1)
        
        self.history.append('get_field_time')
        return self.Et

    def get_phase_time(self, filter_=FILTER_PHASE):
        """
        :param filter_: whether to filter phase values with low amplitude
        :return: beam temporal phase profile across its cross-section after removing the carrier frequency
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'get_phase_time':
                return self.phase_t
            elif method in LOG.MODIFIERS:
                break
            index -= 1
        
        Et = self.get_field_time()
        φt = np.arctan2(Et.imag, Et.real)
        self.phase_t = np.unwrap(φt, axis=1)
        for j in np.arange(self.Nx):
            self.phase_t[j] += φt[j][self.Nf // 2] - self.phase_t[j][self.Nf // 2]  # unwrapping should not change center phase
        if filter_:
            self.phase_t = np.where(np.abs(Et) > np.amax(np.abs(Et)) * ε, self.phase_t, np.nan)

        self.history.append('get_phase_time')
        return self.phase_t

    def get_amplitude_time(self):
        """
        :return: beam temporal amplitude profile across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'get_amplitude_time':
                return self.At
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        self.At = np.abs(self.get_field_time())
        
        self.history.append('get_amplitude_time')
        return self.At

    def get_intensity_time(self):
        """
        :return: beam temporal intensity profile across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'get_intensity_time':
                return self.It
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        self.It = np.abs(self.get_field_time()) ** 2 / (2 * μ0)
        
        self.history.append('get_intensity_time')
        return self.It

    def get_field_frequency(self):
        """
        :return: beam complex field values in frequency across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'get_field_frequency':
                return self.Ef
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        self.Ef = np.transpose([b.get_field() for b in self.beams])
        
        self.history.append('get_field_frequency')
        return self.Ef

    def get_phase_frequency(self, filter_=FILTER_PHASE):
        """
        :param filter_: whether to filter values with low amplitude
        :return: beam spectral phase profile across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'get_phase_frequency':
                return self.phase_f
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        φf = np.transpose([b.get_phase() for b in self.beams])
        self.phase_f = np.unwrap(φf, axis=1)
        for j in np.arange(self.Nx):
            self.phase_f[j] += φf[j][self.Nf // 2] - self.phase_f[j][self.Nf // 2]  # unwrapping should not change center phase
        if filter_:
            Af = self.get_amplitude_frequency()
            self.phase_f = np.where(Af > np.amax(Af) * ε, self.phase_f, np.nan)
            
        self.history.append('get_phase_frequency')
        return self.phase_f

    def get_amplitude_frequency(self):
        """
        :return: beam spectral amplitude profile across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'get_amplitude_frequency':
                return self.Af
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        self.Af = np.transpose([b.get_amplitude() for b in self.beams])
        
        self.history.append('get_amplitude_frequency')
        return self.Af

    def get_intensity_frequency(self):
        """
        :return: beam spectral intensity profile across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'get_intensity_frequency':
                return self.If
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        self.If = np.transpose([b.get_intensity() for b in self.beams])
        
        self.history.append('get_intensity_frequency')
        return self.If

    def get_wigner(self):
        """
        :return: Wigner distribution of the central beam at position x=0
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'get_wigner':
                return self.W
            elif method in LOG.MODIFIERS:
                break
            index -= 1

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

        self.history.append('get_wigner')
        return self.W

    def plot_field_time(self, title=PLOT.FIELD.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam real field values in time after removing the carrier frequency.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central field values
        :return: PolyBeam object for chaining
        """
        values = np.real(self.get_field_time())
        self._plot(PLOT.FIELD, PLOT.TIME, values, self.ts, title, cmap)
        return self

    def plot_phase_time(self, title=PLOT.PHASE.TITLE, cmap=PLOT_CMAP, filter_=FILTER_PHASE):
        """
        Plot beam temporal phase profile after subtracting entral carrier frequency.
        :param title: title of the plot
        :param filter_: whether to filter values with low amplitude
        :param cmap: whether to plot a color map across beam cross-section or only the central phase values
        :return: PolyBeam object for chaining
        """
        values = self.get_phase_time(filter_=filter_)
        self._plot(PLOT.PHASE, PLOT.TIME, values, self.ts, title, cmap)
        return self

    def plot_amplitude_time(self, title=PLOT.AMPLITUDE.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam temporal amplitude profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central amplitude values
        :return: PolyBeam object for chaining
        """
        values = self.get_amplitude_time()
        self._plot(PLOT.AMPLITUDE, PLOT.TIME, values, self.ts, title, cmap)
        return self

    def plot_intensity_time(self, title=PLOT.INTENSITY.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam temporal intensity profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central intensity values
        :return: PolyBeam object for chaining
        """
        values = self.get_intensity_time()
        self._plot(PLOT.INTENSITY, PLOT.TIME, values, self.ts, title, cmap)
        return self

    def plot_field_frequency(self, title=PLOT.FIELD.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam real field values in frequency.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central field values
        :return: PolyBeam object for chaining
        """
        values = np.real(self.get_field_frequency())
        self._plot(PLOT.FIELD, PLOT.FREQUENCY, values, self.fs, title, cmap)
        return self

    def plot_phase_frequency(self, title=PLOT.PHASE.TITLE, cmap=PLOT_CMAP, filter_=FILTER_PHASE):
        """
        Plot beam spectral phase profile.
        :param title: title of the plot
        :param filter_: whether to filter values with low amplitude
        :param cmap: whether to plot a color map across beam cross-section or only the central phase values
        :return: PolyBeam object for chaining
        """
        values = self.get_phase_frequency(filter_=filter_)
        self._plot(PLOT.PHASE, PLOT.FREQUENCY, values, self.fs, title, cmap)
        return self

    def plot_amplitude_frequency(self, title=PLOT.AMPLITUDE.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam spectral amplitude profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central amplitude values
        :return: PolyBeam object for chaining
        """
        values = self.get_amplitude_frequency()
        self._plot(PLOT.AMPLITUDE, PLOT.FREQUENCY, values, self.fs, title, cmap)
        return self

    def plot_intensity_frequency(self, title=PLOT.INTENSITY.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam spectral intensity profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central intensity values
        :return: PolyBeam object for chaining
        """
        values = self.get_intensity_frequency()
        self._plot(PLOT.INTENSITY, PLOT.FREQUENCY, values, self.fs, title, cmap)
        return self

    def plot_wigner(self, title=PLOT.WIGNER.TITLE):
        """
        Plot the Wigner distribution of the central beam at position x=0.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        with color_palette('husl'):
            Rt = (self.ts[0] * PLOT.TIME.SCALE, self.ts[-1] * PLOT.TIME.SCALE)
            Rf = (self.fs[0] * PLOT.FREQUENCY.SCALE, self.fs[-1] * PLOT.FREQUENCY.SCALE)
            W = self.get_wigner() * PLOT.WIGNER.SCALE
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
            area_W.set_xlabel(PLOT.TIME.LABEL)
            area_W.set_ylabel(PLOT.FREQUENCY.LABEL)
            plt.colorbar(cmap, ax=area_main).set_label(PLOT.INTENSITY.LABEL)

            area_If.plot(If, self.fs * PLOT.FREQUENCY.SCALE, 'b')
            area_If.margins(y=0)
            area_If.set_xticks([])
            area_If.set_yticks([])

            area_It.plot(self.ts * PLOT.TIME.SCALE, It, 'b')
            area_It.margins(x=0)
            area_It.set_xticks([])
            area_It.set_yticks([])

            plt.gcf().canvas.set_window_title('-'.join(title.lower().split()))
            with warnings.catch_warnings():  # matplotlib throws a UserWarning that I can't fix!
                warnings.simplefilter('ignore')
                plt.show()
        return self

    def _plot_line(self, values, xs, title, label, versus):  # todo: fix non-matching x positions
        if any([np.isfinite(b.Rp) for b in self.beams]):
            raise RuntimeError(ERROR.PLOT_NONPLANAR)

        versus_values = xs * versus.SCALE
        plt.plot(versus_values, values)
        plt.xlim(versus_values[0], versus_values[-1])
        plt.title(title)
        plt.gcf().canvas.set_window_title('-'.join(title.lower().split() + [versus.TITLE.lower()]))
        plt.xlabel(versus.LABEL)
        plt.ylabel(label)
        plt.show()

    def _plot_cmap(self, values, xs, title, label, versus):
        with color_palette('husl'):
            if any([np.isfinite(b.Rp) for b in self.beams]):
                raise RuntimeError(ERROR.PLOT_NONPLANAR)

            versus_values = xs * versus.SCALE
            wx = self.Dx / 2 * PLOT.POSITION.SCALE

            plt.imshow(values, aspect='auto', extent=[versus_values[0], versus_values[-1], -wx, wx],
                       interpolation='none')
            plt.title(title)
            plt.gcf().canvas.set_window_title('-'.join(title.lower().split() + [versus.TITLE.lower()]))
            plt.xlabel(versus.LABEL)
            plt.ylabel(PLOT.POSITION.LABEL)
            plt.colorbar().set_label(label)
            plt.show()

    def _plot(self, variable, versus, values, xs, title, cmap):
        if cmap:
            vs = values * variable.SCALE
            self._plot_cmap(vs, xs, title, variable.LABEL, versus)
        else:
            vs = values[self.Nx // 2] * variable.SCALE
            self._plot_line(vs, xs, title, variable.LABEL, versus)
