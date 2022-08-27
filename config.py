import contextlib
import pathlib
from os import path
import pyfftw
import pickle
import dataclasses

i = complex(0, 1)
c = 2.9979E8
π = 3.14159
μ0 = 1.2566E-6
ε = 1E-5  # small threshold value to filter phases with small amplitude
ηR = 2  # todo: should be set dynamically?
δz = 1E-4

ASSETS_DIR = pathlib.Path(path.dirname(__file__)) / 'assets'
WISDOM_FILE = 'pyfftw_wisdom_{}'
VERBOSE = True  # whether to print beam operations and their runtimes
FILTER_PHASE = True  # whether to filter low amplitude values for phase profile by default
PLOT_CMAP = True  # whether to plot color maps by default


class PRE:
    T = 1E12
    G = 1E9
    c = 1E-2
    m = 1E-3
    p = 1E-12
    f = 1E-15


@dataclasses.dataclass
class Variable:
    TITLE: str
    LABEL: str
    SCALE: float


class ERROR:
    ROTATE_ALIAS = 'Rotation causes phase aliasing. Make any of the following changes to relieve:\n' \
                   '|α| < {:.1f}° or Nx >= 2^{:.0f} or ηx < {:.1f}.'
    TRANSFORM_NONPLANAR = 'Transform failed since some of the cross-sections are not planar.'
    PLOT_NONPLANAR = 'Cannot plot since some of the cross-sections are not planar.'
    DISPERSE = 'Some frequency components don\'t have a first-order diffraction for these grating parameters.'
    CHIRP_ALIAS = 'Chirp causes phase aliasing. Make any of the following changes to relieve:\n' \
                  'r < {:.0f}ps² or Nf >= 2^{:.0f} or ηf < {:.1f}.'


class PROPAGATOR:
    IN_TO_IN = (True, True)
    IN_TO_OUT = (True, False)
    OUT_TO_IN = (False, True)
    OUT_TO_OUT = (False, False)


class LOG:
    MODIFIERS = {'rotate', 'mask', 'propagate', 'lens', 'disperse', 'chirp'}

    INIT = 'Initialize f0={:.0f}THz, Δf={:.1f}THz, Δx={:.1f}mm'
    ROTATE = 'Rotate α={:.1f}°'
    MASK = 'Apply mask'
    PROPAGATE = 'Propagate Δz={:.1f}cm'
    LENS = 'Apply lens f={:.1f}cm'
    DISPERSE = 'Disperse d={:d}mm⁻¹, α={:.1f}°'
    CHIRP = 'Add chirp r={:.3f}ps²'
    TIME = '[{:.2f}s].'


class PLOT:
    POSITION = Variable('Position', 'x (mm)', 1 / PRE.m)
    FREQUENCY = Variable('Frequency', 'f (THz)', 1 / PRE.T)
    TIME = Variable('Time', 't (ps)', 1 / PRE.p)

    PHASE = Variable('Phase Profile', 'φ (rad)', 1)
    AMPLITUDE = Variable('Amplitude Profile', 'A (V/m)', 1)
    FIELD = Variable('Field Values', 'E (V/m)', 1)
    INTENSITY = Variable('Intensity Profile', 'I (GW/m²)', 1 / PRE.G)
    WIGNER = Variable('Wigner Distribution', '', 1)

    PLANAR_WARN = 'Plotting "{}" on planar cross-section.'
    SPHERICAL_WARN = 'Plotting "{}" on spherical cross-section with curvature {:.1f}cm.'


@contextlib.contextmanager
def pyfftw_wisdom(wisdom_path):
    if path.exists(wisdom_path):
        with open(wisdom_path, 'rb') as file:
            pyfftw.import_wisdom(pickle.load(file))
        yield
    else:
        yield
        with open(wisdom_path, 'wb') as file:
            pickle.dump(pyfftw.export_wisdom(), file, 2)
