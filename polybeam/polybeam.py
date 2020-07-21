import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette
from monobeam import MonoBeam
from polybeam_config import *
import pyfftw
import multiprocessing
import pickle
import time
import contextlib
import gc


class PolyBeam:
    def __init__(self, ν0, Δν, Δx, Dν=None, Nν=None, Dx=None, Nx=None, I0=None, do_log=True):
        """
        Specify a polychromatic input beam with a Gaussian cross-sectional profile and spectral envelope.
        :param ν0: central frequency of the beam (THz)
        :param Δν: FWHM of the spectral intensity (THz)
        :param Δx: FWHM of the cross-sectional intensity (mm)
        :param Dν: width of the spectral sampling region (THz)
        :param Nν: number of points to sample in the beam
        :param Dx: width of the cross-sectional sampling region (mm)
        :param Nx: number of points to sample in the cross-section
        :param I0: beam intensity at its center (TW/m²)
        :param do_log: whether to print beam operations and their runtimes
        """
        self.do_log = do_log
        with self._log(LOG.INIT.format(ν0, Δν, Δx)):
            pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

            self.ν0 = ν0 * PRE.T
            self.Dν = (Dν or ην * Δν) * PRE.T
            self.Nν = Nν or DEFAULT_Nν
            self.Nx = Nx or DEFAULT_Nx
            self.dν = self.Dν / self.Nν
            self.νs = np.linspace(start=self.ν0 - self.Dν / 2, stop=self.ν0 + self.Dν / 2, num=self.Nν)

            self.Et = np.ndarray(shape=(self.Nx, self.Nν), dtype=np.complex128)
            self.history = []
            self.beams = []

            for ν in self.νs:
                ν /= PRE.T
                I = (I0 or DEFAULT_I0) * np.exp(-4 * np.log(2) * ((ν0 - ν) / Δν) ** 2)
                self.beams.append(MonoBeam(ν=ν, Δx=Δx, I0=I, Dx=(Dx or ηx * Δx), Nx=self.Nx))

    def rotate(self, α):
        """
        Rotate the beam by applying a linear phase in position.
        :param α: angle to rotate by (°)
        :return: PolyBeam object for chaining
        """
        with self._log(LOG.ROTATE.format(α)):
            for beam in self.beams:
                beam.rotate(α=α)
        return self

    def mask(self, M):
        """
        Apply a complex mask to modulate the beam amplitude and phase.
        :param M: mask function that maps list of position (mm) to their complex multiplier
        :return: PolyBeam object for chaining
        """
        with self._log(LOG.MASK.format(M)):
            for beam in self.beams:
                beam.mask(M=M)
        return self

    def propagate(self, Δz):
        """
        Propagate the beam in free space.
        :param Δz: distance to propagate by (cm)
        :return: PolyBeam object for chaining
        """
        with self._log(LOG.PROPAGATE.format(Δz)):
            for beam in self.beams:
                beam.propagate(Δz=Δz)
        return self

    def lens(self, f):
        """
        Simulate the effect of a thin lens on the beam by applying a quadratic phase in position.
        :param f: focal length of the lens (cm)
        :return: PolyBeam object for chaining
        """
        with self._log(LOG.LENS.format(f)):
            for beam in self.beams:
                beam.lens(f=f)
        return self

    def disperse(self, d, α):
        """
        Simulate a dispersive first-order diffraction by applying frequency-dependent rotation.
        :param d: grating groove density (1/mm)
        :param α: incident beam angle (°)
        :raises RuntimeError: if first order diffraction does not exist for some frequencies
        :return: PolyBeam object for chaining
        """
        with self._log(LOG.DISPERSE.format(d, α)):
            δ = PRE.m / d
            α = np.deg2rad(α)
            if np.isnan(np.arcsin(c / (self.νs[0] * δ) - np.sin(α)) - np.arcsin(c / (self.νs[-1] * δ) - np.sin(α))):
                raise RuntimeError(ERROR.DISPERSION)
            θ0 = np.arcsin(c / (self.ν0 * δ) - np.sin(α))
            for beam, ν in zip(self.beams, self.νs):
                beam.rotate(α=np.rad2deg(np.arcsin(c / (ν * δ) - np.sin(α)) - θ0))
        return self

    def chirp(self, α):
        """
        Apply a quadratic phase shift in frequency to chirp the beam.
        :param α: the chirp rate (fs²)
        :return: PolyBeam object for chaining
        """
        with self._log(LOG.CHIRP.format(α)):
            α *= PRE.f ** 2
            for ν, beam in zip(self.νs, self.beams):
                m = np.exp(4 * π ** 2 * i * α * (ν - self.ν0) ** 2)
                beam.mask(M=lambda xs: m)
        return self

    def get_transform(self):
        """
        Fourier transform beam from frequency to time domain and centers it.
        :return: beam complex amplitude in time domain for all positions
        :raises RuntimeError: if any of the component beams is not planar
        """
        if self.history[-1] == LOG.TRANSFORM:
            return self.Et

        with self._log(LOG.TRANSFORM):
            if any([np.isfinite(b.Rp) for b in self.beams]):
                raise RuntimeError(ERROR.TRANSFORM)

            for j, beam in enumerate(self.beams):
                self.Et[:, j] = beam.E
            with self._pyfftw_wisdom():
                for j, E in enumerate(self.Et):
                    fftw = pyfftw.FFTW(E, E, direction='FFTW_BACKWARD', flags=['FFTW_UNALIGNED', 'FFTW_ESTIMATE'])
                    self.Et[j] = fftw() * np.sqrt(self.Nν)
            self.Et = np.roll(np.roll(self.Et, self.Nν // 2 - np.argmax(self.Et) % self.Nν), self.Nx // 2, axis=0)
        return self.Et

    def get_phase_time(self, do_filter=True):
        """
        Get beam temporal phase profile for its cross-section after subtracting carrier frequencies.
        :param do_filter: whether to filter phase values with low amplitude
        :return: PolyBeam object for chaining
        """
        Et = self.get_transform()
        φs = np.arctan2(Et.imag, Et.real)
        φsu = np.unwrap(φs, axis=1)
        for j in np.arange(self.Nx):
            φsu[j] += φs[j][self.Nν // 2] - φsu[j][self.Nν // 2]  # unwrapping should leave center phase unchanged

        # cs = np.polyfit(np.arange(self.Nν), φsu[self.Nx // 2], 1)  # subtract linear changes
        # for j in np.arange(self.Nx):
        #     φsu[j] -= cs[1] + cs[0] * np.arange(self.Nν)

        if do_filter:
            φsu = np.where(np.abs(Et) > np.amax(np.abs(Et)) * ςE, φsu, np.nan)
        return φsu

    def get_amplitude_time(self):
        """
        Get beam temporal amplitude profile for its cross-section.
        :return: PolyBeam object for chaining
        """
        return np.abs(self.get_transform())

    def get_intensity_time(self):
        """
        Get beam temporal intensity profile for its cross-section.
        :return: PolyBeam object for chaining
        """
        return np.abs(self.get_transform()) ** 2 / (2 * μ0)

    def get_phase_freq(self, do_filter=True):
        """
        Get beam spectral phase profile for its cross-section.
        :param do_filter: whether to filter values with low amplitude
        :return: PolyBeam object for chaining
        """
        φs = [b.get_phase_section() for b in self.beams]
        if do_filter:
            As = [b.get_amplitude_section() for b in self.beams]
            φs = np.where(As > np.amax(As) * ςE, φs, np.nan)
        return φs

    def get_amplitude_freq(self):
        """
        Get beam spectral amplitude profile for its cross-section.
        :return: PolyBeam object for chaining
        """
        return [b.get_amplitude_section() for b in self.beams]

    def get_intensity_freq(self):
        """
        Get beam spectral intensity profile for its cross-section.
        :return: PolyBeam object for chaining
        """
        return [b.get_intensity_section() for b in self.beams]

    def get_wigner(self):
        """
        Get the Wigner distribution of the central beam at position x=0.
        :return: PolyBeam object for chaining
        """
        Et = np.pad(self.get_transform()[self.Nx // 2], pad_width=self.Nν, constant_values=0)
        W = np.ndarray(shape=(self.Nν, self.Nν), dtype=np.complex128)

        for t in range(self.Nν):
            for s in range(self.Nν):
                W[t][s] = Et[t + s + self.Nν] * np.conj(Et[t - s + self.Nν]) / (2 * μ0)

        with self._pyfftw_wisdom():
            for t, Wt in enumerate(W):
                fftw = pyfftw.FFTW(Wt, Wt, direction='FFTW_FORWARD', flags=['FFTW_UNALIGNED', 'FFTW_ESTIMATE'])
                W[t] = fftw() / np.sqrt(self.Nν)
        return np.abs(np.roll(np.transpose(W[self.Nν // 4: 3 * self.Nν // 4]), self.Nν // 2, axis=0))

    def plot_phase_time(self, title=PLOT.PHASE_TITLE, do_filter=True):
        """
        Plot beam temporal phase profile for its cross-section after subtracting carrier frequencies.
        :param title: title of the plot
        :param do_filter: whether to filter values with low amplitude
        :return: PolyBeam object for chaining
        """
        self._plot_time(vs=self.get_phase_time(do_filter=do_filter), title=title, label=PLOT.PHASE_LABEL)
        return self

    def plot_amplitude_time(self, title=PLOT.AMPLITUDE_TITLE):
        """
        Plot beam temporal amplitude profile for its cross-section.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        self._plot_time(vs=self.get_amplitude_time() * PLOT.AMPLITUDE_SCALE, title=title, label=PLOT.AMPLITUDE_LABEL)
        return self

    def plot_intensity_time(self, title=PLOT.AMPLITUDE_TITLE):
        """
        Plot beam temporal intensity profile for its cross-section.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        self._plot_time(vs=self.get_intensity_time() * PLOT.INTENSITY_SCALE, title=title, label=PLOT.INTENSITY_LABEL)
        return self

    def plot_phase_freq(self, title=PLOT.PHASE_TITLE, do_filter=True):
        """
        Plot beam spectral phase profile for its cross-section.
        :param title: title of the plot
        :param do_filter: whether to filter values with low amplitude
        :return: PolyBeam object for chaining
        """
        self._plot_freq(vs=self.get_phase_freq(do_filter=do_filter), title=title, label=PLOT.PHASE_LABEL)
        return self

    def plot_amplitude_freq(self, title=PLOT.AMPLITUDE_TITLE):
        """
        Plot beam spectral amplitude profile for its cross-section.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        self._plot_freq(vs=self.get_amplitude_freq() * PLOT.AMPLITUDE_SCALE, title=title, label=PLOT.AMPLITUDE_LABEL)
        return self

    def plot_intensity_freq(self, title=PLOT.AMPLITUDE_TITLE):
        """
        Plot beam spectral intensity profile for its cross-section.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        self._plot_freq(vs=self.get_intensity_freq() * PLOT.INTENSITY_SCALE, title=title, label=PLOT.INTENSITY_LABEL)
        return self

    def plot_wigner(self, title=PLOT.WIGNER_TITLE):
        """
        Plot the Wigner distribution of the central beam at position x=0.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        with color_palette('husl'):
            wt = 1 / (2 * self.dν) * PLOT.TIME_SCALE
            Rν = (self.νs[0] * PLOT.FREQUENCY_SCALE, self.νs[-1] * PLOT.FREQUENCY_SCALE)

            plt.imshow(self.get_wigner() * PLOT.INTENSITY_SCALE, aspect='auto', extent=[-wt, wt, *Rν])
            plt.gcf().canvas.set_window_title(PLOT.SAVE_WIGNER(title))
            plt.title(title)
            plt.xlabel(xlabel=PLOT.TIME_LABEL)
            plt.ylabel(ylabel=PLOT.FREQUENCY_LABEL)
            plt.colorbar().set_label(PLOT.INTENSITY_LABEL)
            plt.show()
        return self

    def _plot_freq(self, vs, title, label):  # todo: fix the non-matching x positions
        with color_palette('husl'):
            if any([np.isfinite(b.Rp) for b in self.beams]):
                raise RuntimeError(ERROR.SECTION)

            wx = self.beams[0].dx * self.Nx / 2 * PLOT.POSITION_SCALE
            Rν = (self.νs[0] * PLOT.FREQUENCY_SCALE, self.νs[-1] * PLOT.FREQUENCY_SCALE)

            plt.imshow(np.transpose(vs), aspect='auto', extent=[*Rν, -wx, wx])
            plt.title(title)
            plt.gcf().canvas.set_window_title(PLOT.SAVE_FREQ(title))
            plt.xlabel(xlabel=PLOT.FREQUENCY_LABEL)
            plt.ylabel(ylabel=PLOT.POSITION_LABEL)
            plt.colorbar().set_label(label)
            plt.show()

    def _plot_time(self, vs, title, label):
        with color_palette('husl'):
            if any([np.isfinite(b.Rp) for b in self.beams]):
                raise RuntimeError(ERROR.SECTION)

            wt = 1 / (2 * self.dν) * PLOT.TIME_SCALE
            wx = self.beams[0].dx * self.Nx / 2 * PLOT.POSITION_SCALE

            plt.imshow(vs, aspect='auto', extent=[-wt, wt, -wx, wx])
            plt.gcf().canvas.set_window_title(PLOT.SAVE_TIME(title))
            plt.title(title)
            plt.xlabel(xlabel=PLOT.TIME_LABEL)
            plt.ylabel(ylabel=PLOT.POSITION_LABEL)
            plt.colorbar().set_label(label)
            plt.show()

    @contextlib.contextmanager
    def _pyfftw_wisdom(self):
        wisdom_path = ASSETS_DIR / WISDOM_FILE.format(self.Nν)
        if path.exists(wisdom_path):
            with open(wisdom_path, 'rb') as file:
                pyfftw.import_wisdom(pickle.load(file))
            yield
        else:
            yield
            with open(wisdom_path, 'wb') as file:
                pickle.dump(pyfftw.export_wisdom(), file, 2)

    @contextlib.contextmanager
    def _log(self, message):
        if self.do_log:
            print(message + LOG.WAIT, end='')
            start_time = time.time()
            yield
            print(LOG.TIME.format(time.time() - start_time))
        else:
            yield
        self.history.append(message)
        gc.collect()
