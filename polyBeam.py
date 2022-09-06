import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette
from monoBeam import MonoBeam
from config import *
import pyfftw
import multiprocessing
import time
import gc
import warnings


DEFAULT_ηf = 16
DEFAULT_ηx = 4
DEFAULT_Nf = 2 ** 9
DEFAULT_Nx = 2 ** 12
DEFAULT_I0 = 1  # (TW/m²)


class PolyBeam:
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
        startTime = time.time()
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
        
        if VERBOSE:
            print(LOG.INIT.format(f0, Δf, Δx), LOG.TIME.format(time.time() - startTime))
        self.history.append('init')
        gc.collect()

    def rotate(self, α):
        """
        Rotate the beam by applying a linear phase in position.
        :param α: angle to rotate by (°)
        :return: PolyBeam object for chaining
        """
        startTime = time.time()
        
        for beam in self.beams:
            beam.rotate(α=α)
        
        if VERBOSE:
            print(LOG.ROTATE.format(α), LOG.TIME.format(time.time() - startTime))
        self.history.append('rotate')
        gc.collect()
        return self

    def mask(self, M):  # todo: fix mask and rotate for non-planar
        """
        Apply a complex mask to modulate the beam amplitude and phase.
        :param M: mask function that maps list of position (mm) to their complex multiplier
        :return: PolyBeam object for chaining
        """
        startTime = time.time()

        for beam in self.beams:
            beam.mask(M=M)

        if VERBOSE:
            print(LOG.MASK, LOG.TIME.format(time.time() - startTime))
        self.history.append('mask')
        gc.collect()
        return self

    def propagate(self, Δz):
        """
        Propagate the beam in free space.
        :param Δz: distance to propagate by (cm)
        :return: PolyBeam object for chaining
        """
        startTime = time.time()
        
        for beam in self.beams:
            beam.propagate(Δz=Δz)
        self.Dx = self.beams[self.Nf // 2].Dx
        
        if VERBOSE:
            print(LOG.PROPAGATE.format(Δz), LOG.TIME.format(time.time() - startTime))
        self.history.append('propagate')
        gc.collect()
        return self

    def lens(self, f):
        """
        Simulate the effect of a thin lens on the beam by applying a quadratic phase in position.
        :param f: focal length of the lens (cm)
        :return: PolyBeam object for chaining
        """
        startTime = time.time()

        for beam in self.beams:
            beam.lens(f=f)

        if VERBOSE:
            print(LOG.LENS.format(f), LOG.TIME.format(time.time() - startTime))
        self.history.append('lens')
        gc.collect()
        return self

    def disperse(self, d, α):
        """
        Simulate a dispersive first-order diffraction by applying frequency-dependent rotation.
        :param d: grating groove density (1/mm)
        :param α: incident beam angle (°)
        :raises RuntimeError: if first order diffraction does not exist for some frequencies
        :raises RuntimeError: if phase aliasing occurs upon dispersion
        :return: PolyBeam object for chaining
        """
        startTime = time.time()

        δ = PRE.m / d
        α = np.deg2rad(α)
        if np.isnan(np.arcsin(c / (self.fs[0] * δ) - np.sin(α)) - np.arcsin(c / (self.fs[-1] * δ) - np.sin(α))):
            raise RuntimeError(ERROR.DISPERSE)
        θ0 = np.arcsin(c / (self.f0 * δ) - np.sin(α))
        for beam, f in zip(self.beams, self.fs):
            beam.rotate(α=np.rad2deg(np.arcsin(c / (f * δ) - np.sin(α)) - θ0))

        if VERBOSE:
            print(LOG.DISPERSE.format(d, α), LOG.TIME.format(time.time() - startTime))
        self.history.append('disperse')
        gc.collect()
        return self

    def chirp(self, r):
        """
        Apply a quadratic phase shift in frequency to chirp the beam.
        :param r: the chirp rate (ps²)
        :return: PolyBeam object for chaining
        """
        startTime = time.time()

        r *= PRE.p ** 2
        if self.Nf < 4 * π * r * self.Df ** 2:  # require: 4πα Df^2 < Nf
            raise RuntimeError(ERROR.CHIRP_ALIAS.format(
                self.Nf / (4 * π * self.Df ** 2) / PRE.f ** 2,
                np.ceil(np.log2(4 * π * r * self.Df ** 2)),
                np.sqrt(self.Nf / (4 * π * r)) / self.Δf))

        for f, beam in zip(self.fs, self.beams):
            m = np.exp(4 * π ** 2 * i * r * (f - self.f0) ** 2)
            beam.mask(M=lambda xs: m)

        if VERBOSE:
            print(LOG.CHIRP.format(r / PRE.p ** 2), LOG.TIME.format(time.time() - startTime))
        self.history.append('chirp')
        gc.collect()
        return self

    def getFieldTime(self):
        """
        Fourier transform beam from frequency to time domain and center it.
        :return: beam complex field values in time across its cross-section after removing the carrier frequency
        :raises RuntimeError: if any of the component beams is not planar
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'getFieldTime':
                return self.Et
            elif method in LOG.MODIFIERS:
                break
            index -= 1
        
        if any([np.isfinite(b.Rp) for b in self.beams]):
            raise RuntimeError(ERROR.TRANSFORM_NONPLANAR)

        self.Et = np.roll(self.getFieldFrequency(), self.Nf // 2, axis=1)
        with pyfftwWisdom(ASSETS_DIR / WISDOM_FILE.format(self.Nf)):
            for j, E in enumerate(self.Et):
                fftw = pyfftw.FFTW(E, E, direction='FFTW_BACKWARD', flags=['FFTW_UNALIGNED', 'FFTW_ESTIMATE'])
                self.Et[j] = fftw() * np.sqrt(self.Nf)
        self.Et = np.roll(self.Et, self.Nf // 2, axis=1)
        
        self.history.append('getFieldTime')
        return self.Et

    def getPhaseTime(self, filter_=FILTER_PHASE):
        """
        :param filter_: whether to filter phase values with low amplitude
        :return: beam temporal phase profile across its cross-section after removing the carrier frequency
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'getPhaseTime':
                return self.φt
            elif method in LOG.MODIFIERS:
                break
            index -= 1
        
        Et = self.getFieldTime()
        φt = np.arctan2(Et.imag, Et.real)
        self.φt = np.unwrap(φt, axis=1)
        for j in np.arange(self.Nx):
            self.φt[j] += φt[j][self.Nf // 2] - self.φt[j][self.Nf // 2]  # unwrapping should not change center phase
        if filter_:
            self.φt = np.where(np.abs(Et) > np.amax(np.abs(Et)) * ε, self.φt, np.nan)

        self.history.append('getPhaseTime')
        return self.φt

    def getAmplitudeTime(self):
        """
        :return: beam temporal amplitude profile across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'getAmplitudeTime':
                return self.At
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        self.At = np.abs(self.getFieldTime())
        
        self.history.append('getAmplitudeTime')
        return self.At

    def getIntensityTime(self):
        """
        :return: beam temporal intensity profile across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'getIntensityTime':
                return self.It
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        self.It = np.abs(self.getFieldTime()) ** 2 / (2 * μ0)
        
        self.history.append('getIntensityTime')
        return self.It

    def getFieldFrequency(self):
        """
        :return: beam complex field values in frequency across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'getFieldFrequency':
                return self.Ef
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        self.Ef = np.transpose([b.getField() for b in self.beams])
        
        self.history.append('getFieldFrequency')
        return self.Ef

    def getPhaseFrequency(self, filter_=FILTER_PHASE):
        """
        :param filter_: whether to filter values with low amplitude
        :return: beam spectral phase profile across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'getPhaseFrequency':
                return self.φf
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        φf = np.transpose([b.getPhase() for b in self.beams])
        self.φf = np.unwrap(φf, axis=1)
        for j in np.arange(self.Nx):
            self.φf[j] += φf[j][self.Nf // 2] - self.φf[j][self.Nf // 2]  # unwrapping should not change center phase
        if filter_:
            Af = self.getAmplitudeFrequency()
            self.φf = np.where(Af > np.amax(Af) * ε, self.φf, np.nan)
            
        self.history.append('getPhaseFrequency')
        return self.φf

    def getAmplitudeFrequency(self):
        """
        :return: beam spectral amplitude profile across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'getAmplitudeFrequency':
                return self.Af
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        self.Af = np.transpose([b.getAmplitude() for b in self.beams])
        
        self.history.append('getAmplitudeFrequency')
        return self.Af

    def getIntensityFrequency(self):
        """
        :return: beam spectral intensity profile across its cross-section
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'getIntensityFrequency':
                return self.If
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        self.If = np.transpose([b.getIntensity() for b in self.beams])
        
        self.history.append('getIntensityFrequency')
        return self.If

    def getWigner(self):
        """
        :return: Wigner distribution of the central beam at position x=0
        """
        index = len(self.history) - 1
        while index >= 0:
            method = self.history[index]
            if method == 'getWigner':
                return self.W
            elif method in LOG.MODIFIERS:
                break
            index -= 1

        E = self.getFieldTime()[self.Nx // 2]
        Et = np.pad(E, (0, self.Nf))
        Es = np.pad(np.flip(np.conj(E)), (0, self.Nf))
        self.W = np.ndarray(shape=(self.Nf, self.Nf), dtype=np.complex128)

        for t in range(self.Nf):
            self.W[t] = np.multiply(Et[t: t + self.Nf], Es[self.Nf - t - 1: 2 * self.Nf - t - 1]) / (2 * μ0)

        with pyfftwWisdom(ASSETS_DIR / WISDOM_FILE.format(self.Nf)):
            for t, Wt in enumerate(self.W):
                fftw = pyfftw.FFTW(Wt, Wt, direction='FFTW_FORWARD', flags=['FFTW_UNALIGNED', 'FFTW_ESTIMATE'])
                self.W[t] = fftw() / np.sqrt(self.Nf)
        self.W = np.roll(self.W.T, self.Nf // 2, axis=0)

        self.history.append('getWigner')
        return self.W

    def plotFieldTime(self, title=PLOT.FIELD.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam real field values in time after removing the carrier frequency.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central field values
        :return: PolyBeam object for chaining
        """
        values = np.real(self.getFieldTime())
        self._plot(PLOT.FIELD, PLOT.TIME, values, self.ts, title, cmap)
        return self

    def plotPhaseTime(self, title=PLOT.PHASE.TITLE, cmap=PLOT_CMAP, filter_=FILTER_PHASE):
        """
        Plot beam temporal phase profile after subtracting entral carrier frequency.
        :param title: title of the plot
        :param filter_: whether to filter values with low amplitude
        :param cmap: whether to plot a color map across beam cross-section or only the central phase values
        :return: PolyBeam object for chaining
        """
        values = self.getPhaseTime(filter_=filter_)
        self._plot(PLOT.PHASE, PLOT.TIME, values, self.ts, title, cmap)
        return self

    def plotAmplitudeTime(self, title=PLOT.AMPLITUDE.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam temporal amplitude profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central amplitude values
        :return: PolyBeam object for chaining
        """
        values = self.getAmplitudeTime()
        self._plot(PLOT.AMPLITUDE, PLOT.TIME, values, self.ts, title, cmap)
        return self

    def plotIntensityTime(self, title=PLOT.INTENSITY.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam temporal intensity profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central intensity values
        :return: PolyBeam object for chaining
        """
        values = self.getIntensityTime()
        self._plot(PLOT.INTENSITY, PLOT.TIME, values, self.ts, title, cmap)
        return self

    def plotFieldFrequency(self, title=PLOT.FIELD.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam real field values in frequency.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central field values
        :return: PolyBeam object for chaining
        """
        values = np.real(self.getFieldFrequency())
        self._plot(PLOT.FIELD, PLOT.FREQUENCY, values, self.fs, title, cmap)
        return self

    def plotPhaseFrequency(self, title=PLOT.PHASE.TITLE, cmap=PLOT_CMAP, filter_=FILTER_PHASE):
        """
        Plot beam spectral phase profile.
        :param title: title of the plot
        :param filter_: whether to filter values with low amplitude
        :param cmap: whether to plot a color map across beam cross-section or only the central phase values
        :return: PolyBeam object for chaining
        """
        values = self.getPhaseFrequency(filter_=filter_)
        self._plot(PLOT.PHASE, PLOT.FREQUENCY, values, self.fs, title, cmap)
        return self

    def plotAmplitudeFrequency(self, title=PLOT.AMPLITUDE.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam spectral amplitude profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central amplitude values
        :return: PolyBeam object for chaining
        """
        values = self.getAmplitudeFrequency()
        self._plot(PLOT.AMPLITUDE, PLOT.FREQUENCY, values, self.fs, title, cmap)
        return self

    def plotIntensityFrequency(self, title=PLOT.INTENSITY.TITLE, cmap=PLOT_CMAP):
        """
        Plot beam spectral intensity profile.
        :param title: title of the plot
        :param cmap: whether to plot a color map across beam cross-section or only the central intensity values
        :return: PolyBeam object for chaining
        """
        values = self.getIntensityFrequency()
        self._plot(PLOT.INTENSITY, PLOT.FREQUENCY, values, self.fs, title, cmap)
        return self

    def plotWigner(self, title=PLOT.WIGNER.TITLE):
        """
        Plot the Wigner distribution of the central beam at position x=0.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        with color_palette('husl'):
            Rt = (self.ts[0] * PLOT.TIME.SCALE, self.ts[-1] * PLOT.TIME.SCALE)
            Rf = (self.fs[0] * PLOT.FREQUENCY.SCALE, self.fs[-1] * PLOT.FREQUENCY.SCALE)
            W = self.getWigner() * PLOT.WIGNER.SCALE
            It = np.abs(np.sum(W, axis=0))
            If = np.abs(np.sum(W, axis=1))

            fig = plt.figure()
            areaMain = fig.add_axes([0.1, 0.12, 0.86, 0.8])
            areaMain.set_title(title)
            areaMain.set_axis_off()

            gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
                                  left=0.12, right=0.81, bottom=0.12, top=0.9, wspace=0.02, hspace=0.02)
            areaW = fig.add_subplot(gs[1, 0])
            areaIt = fig.add_subplot(gs[0, 0])
            areaIf = fig.add_subplot(gs[1, 1])

            cmap = areaW.imshow(np.abs(W), aspect='auto', extent=[*Rt, *Rf], interpolation='none')
            areaW.set_xlabel(PLOT.TIME.LABEL)
            areaW.set_ylabel(PLOT.FREQUENCY.LABEL)
            plt.colorbar(cmap, ax=areaMain).set_label(PLOT.INTENSITY.LABEL)

            areaIf.plot(If, np.flip(self.fs) * PLOT.FREQUENCY.SCALE, 'b')
            areaIf.margins(y=0)
            areaIf.set_xticks([])
            areaIf.set_yticks([])

            areaIt.plot(self.ts * PLOT.TIME.SCALE, It, 'b')
            areaIt.margins(x=0)
            areaIt.set_xticks([])
            areaIt.set_yticks([])

            plt.gcf().canvas.set_window_title('-'.join(title.lower().split()))
            with warnings.catch_warnings():  # matplotlib throws a UserWarning that I can't fix!
                warnings.simplefilter('ignore')
                plt.show()
        return self

    def _plotLine(self, values, xs, title, label, versus):  # todo: fix non-matching x positions
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

    def _plotCmap(self, values, xs, title, label, versus):
        with color_palette('husl'):
            if any([np.isfinite(b.Rp) for b in self.beams]):
                raise RuntimeError(ERROR.PLOT_NONPLANAR)

            versusValues = xs * versus.SCALE
            wx = self.Dx / 2 * PLOT.POSITION.SCALE

            plt.imshow(values, aspect='auto', extent=[versusValues[0], versusValues[-1], -wx, wx],
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
            self._plotCmap(vs, xs, title, variable.LABEL, versus)
        else:
            vs = values[self.Nx // 2] * variable.SCALE
            self._plotLine(vs, xs, title, variable.LABEL, versus)
