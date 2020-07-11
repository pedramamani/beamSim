from monobeam import MonoBeam
import numpy as np
from seaborn import color_palette
import matplotlib.pyplot as plt
from polybeam_config import *
from pyfftw import FFTW, import_wisdom, export_wisdom
from pickle import load, dump


class PolyBeam:
    def __init__(self, ν0, Δν, Δx, Dν=None, Nν=None, Dx=None, Nx=None, I0=None):
        """
        Specify a polychromatic input beam with a Gaussian cross-sectional profile and spectral envelope.
        :param ν0: central frequency of the beam
        :param Δν: FWHM of the spectral intensity
        :param Δx: FWHM of the cross-sectional intensity
        :param Dν: width of the spectral sampling region
        :param Nν: number of points to sample in the beam
        :param Dx: width of the cross-sectional sampling region
        :param Nx: number of points to sample in the cross-section
        :param I0: beam intensity at its center
        """
        Dν = Dν or fν * Δν
        self.Nν = Nν or DEFAULT_Nν
        Dx = Dx or fx * Δx
        self.Nx = Nx or DEFAULT_Nx
        I0 = I0 or DEFAULT_I0
        self.dν = Dν / self.Nν
        self.νs = np.linspace(start=ν0 - Dν / 2, stop=ν0 + Dν / 2, num=self.Nν)

        def intensity(ν): return I0 * np.exp(-4 * np.log(2) * ((ν0 - ν) / Δν) ** 2)
        self.beams = [MonoBeam(ν=ν, Δx=Δx, I0=intensity(ν), Dx=Dx, Nx=self.Nx) for ν in self.νs]

    def rotate(self, α):
        """
        Rotate the beam by applying a linear phase in position.
        :param α: angle to rotate by
        :return: PolyBeam object for chaining
        """
        for beam in self.beams:
            beam.rotate(α=α)
        return self

    def mask(self, M):
        """
        Apply a complex mask to modulate the beam amplitude and phase.
        :param M: mask function that maps a given position to its complex multiplier
        :return: PolyBeam object for chaining
        """
        for beam in self.beams:
            beam.mask(M=M)
        return self

    def propagate(self, Δz):
        """
        Propagate the beam in free space.
        :param Δz: distance to propagate by
        :return: PolyBeam object for chaining
        """
        for beam in self.beams:
            beam.propagate(Δz=Δz)
        return self

    def lens(self, f):
        """
        Simulate the effect of a thin lens on the beam by applying a quadratic phase in position.
        :param f: focal length of the lens (positive/negative for a convex/concave lens)
        :return: PolyBeam object for chaining
        """
        for beam in self.beams:
            beam.lens(f=f)
        return self

    def disperse(self, d, α):
        """
        Simulate a first-order diffraction by applying frequency-dependent rotation.
        :param d: grating spacing
        :param α: incident beam angle
        :raises RuntimeError: if first order diffraction does not exist for some frequencies
        :return: PolyBeam object for chaining
        """
        if np.isnan(np.arcsin(c / (self.νs[0] * d) - np.sin(α)) - np.arcsin(c / (self.νs[-1] * d) - np.sin(α))):
            raise RuntimeError(ERROR.DISPERSION)
        θ0 = np.arcsin(c / (self.νs[self.Nν // 2] * d) - np.sin(α))
        for beam, ν in zip(self.beams, self.νs):
            beam.rotate(α=np.arcsin(c / (ν * d) - np.sin(α)) - θ0)
        return self

    def chirp(self, α):
        """
        Apply a frequency-dependent phase shift to chirp the beam.
        :param α: the chirp rate in units (fs²)
        :return: PolyBeam object for chaining
        """
        α *= 1E-15 ** 2
        for ν, beam in zip(self.νs, self.beams):
            m = np.exp(4 * π ** 2 * i * α * (ν - self.νs[self.Nν // 2]) ** 2)
            beam.mask(M=lambda x: m)
        return self

    def transform(self):
        """
        Fourier transform beam from frequency to time domain and centers it.
        :return: beam complex amplitude in time domain for all positions
        :raises RuntimeError: if any of the component beams is not planar
        """
        if any([np.isfinite(b.Rp) for b in self.beams]):
            raise RuntimeError(ERROR.TRANSFORM)

        wisdom_path = ASSETS_DIR / WISDOM_FILE.format(self.Nν)
        if path.exists(wisdom_path):
            with open(wisdom_path, 'rb') as file:
                import_wisdom(load(file))

        Es = np.ndarray(shape=(self.Nx, self.Nν), dtype=np.complex128)
        for j, beam in enumerate(self.beams):
            Es[:, j] = beam.E
        for j, E in enumerate(Es):
            fftw_object = FFTW(E, E, direction='FFTW_BACKWARD', flags=['FFTW_UNALIGNED', 'FFTW_ESTIMATE'])
            Es[j] = fftw_object() * np.sqrt(self.Nν)

        if not path.exists(wisdom_path):
            with open(wisdom_path, 'wb') as file:
                dump(export_wisdom(), file, 2)
        return np.roll(np.roll(Es, self.Nν // 2 - np.argmax(Es) % self.Nν), self.Nx // 2, axis=0)

    def plot_phase_time(self, title=PLOT.PHASE_TITLE):
        """
        Plot beam temporal phase profile for its cross-section.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        Es = self.transform()
        ϕs = np.arctan2(Es.imag, Es.real)
        ϕsu = np.unwrap(ϕs, axis=0)
        for j in np.arange(self.Nx):
            ϕsu[j] += ϕs[j][self.Nν // 2] - ϕsu[j][self.Nν // 2]  # unwrapping should leave center phase unchanged
        ϕsu = np.unwrap(ϕs, axis=1)
        # cs = np.polyfit(np.arange(self.Nν), ϕsu[self.Nx // 2], 1)
        # for j in np.arange(self.Nx):
        #     ϕsu[j] -= cs[0] + cs[1] * np.arange(self.Nν)
        self._plot_time(vs=ϕsu, title=title, label=PLOT.PHASE_LABEL)
        return self

    def plot_amplitude_time(self, title=PLOT.AMPLITUDE_TITLE):
        """
        Plot beam temporal amplitude profile for its cross-section.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        Es = np.abs(self.transform())
        self._plot_time(vs=Es * PLOT.AMPLITUDE_SCALE, title=title, label=PLOT.AMPLITUDE_LABEL)
        return self

    def plot_intensity_time(self, title=PLOT.AMPLITUDE_TITLE):
        """
        Plot beam temporal intensity profile for its cross-section.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        Is = np.abs(self.transform()) ** 2 / (2 * μ0)
        self._plot_time(vs=Is * PLOT.INTENSITY_SCALE, title=title, label=PLOT.INTENSITY_LABEL)
        return self

    def plot_phase_freq(self, title=PLOT.PHASE_TITLE):
        """
        Plot beam spectral phase profile for its cross-section.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        ϕs = np.unwrap(np.transpose([b.phase_section() for b in self.beams]), axis=1)
        self._plot_section(vs=ϕs, title=title, label=PLOT.PHASE_LABEL)
        return self

    def plot_amplitude_freq(self, title=PLOT.AMPLITUDE_TITLE):
        """
        Plot beam spectral amplitude profile for its cross-section.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        As = np.transpose([b.amplitude_section() for b in self.beams])
        self._plot_section(vs=As * PLOT.AMPLITUDE_SCALE, title=title, label=PLOT.AMPLITUDE_LABEL)
        return self

    def plot_intensity_freq(self, title=PLOT.AMPLITUDE_TITLE):
        """
        Plot beam spectral intensity profile for its cross-section.
        :param title: title of the plot
        :return: PolyBeam object for chaining
        """
        Is = np.transpose([b.intensity_section() for b in self.beams])
        self._plot_section(vs=Is * PLOT.INTENSITY_SCALE, title=title, label=PLOT.INTENSITY_LABEL)
        return self

    def _plot_section(self, vs, title, label):  # todo: fix the non-matching x positions
        with color_palette('husl'):
            if any([np.isfinite(b.Rp) for b in self.beams]):
                raise RuntimeError(ERROR.SECTION)

            wx = self.beams[0].dx * self.Nx / 2 * PLOT.POSITION_SCALE
            Rν = (self.νs[0] * PLOT.FREQUENCY_SCALE, self.νs[-1] * PLOT.FREQUENCY_SCALE)

            plt.imshow(vs, aspect='auto', extent=[*Rν, -wx, wx])
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
