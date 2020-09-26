from pathlib import Path
from os import path
from dataclasses import dataclass

i = complex(0, 1)
c = 2.9979E8
π = 3.14159
μ0 = 1.2566E-6
ε = 1E-5  # small threshold value to filter phases with small amplitude

DEFAULT_ηf = 16
DEFAULT_ηx = 4
DEFAULT_Nf = 2 ** 8
DEFAULT_Nx = 2 ** 12
DEFAULT_I0 = 1  # (TW/m²)

ASSETS_DIR = Path(path.dirname(__file__)) / 'polybeam_assets'
WISDOM_FILE = 'pyfftw_wisdom_{}'

VERBOSE = True  # whether to print beam operations and their runtimes
FILTER_PHASE = True  # whether to filter low amplitude values for phase profile by default
PLOT_CMAP = True  # whether to plot color maps by default


class PRE:  # unit prefixes
    T = 1E12
    G = 1E9
    c = 1E-2
    m = 1E-3
    p = 1E-12
    f = 1E-15


class ERROR:
    TRANSFORM_NONPLANAR = 'Transform failed since some of the cross-sections are not planar.'
    PLOT_NONPLANAR = 'Cannot plot since some of the cross-sections are not planar.'
    DISPERSE = 'Some frequency components don\'t have a first-order diffraction for these grating parameters.'
    CHIRP_ALIAS = 'Chirp causes phase aliasing. Make any of the following changes to relieve:\n' \
                  'α < {:.0f}ps² or Nf >= 2^{:.0f} or ηf < {:.1f}.'


class LOG:
    MODIFIERS = {'rotate', 'mask', 'propagate', 'lens', 'disperse', 'chirp'}

    INIT = 'Initialize f0={:.0f}THz, Δf={:.1f}THz, Δx={:.1f}mm'
    ROTATE = 'Rotate α={:.1f}°'
    MASK = 'Apply mask'
    PROPAGATE = 'Propagate Δz={:.1f}cm'
    LENS = 'Apply lens f={:.1f}cm'
    DISPRESE = 'Disperse d={:d}mm⁻¹, α={:.1f}°'
    CHIRP = 'Add chirp α={:.1f}ps²'
    TIME = '[{:.2f}s].'


class PLOT:
    @dataclass
    class Parameter:
        TITLE: str
        LABEL: str
        SCALE: float

    POSITION = Parameter('Position', 'x (mm)', 1 / PRE.m)
    FREQUENCY = Parameter('Frequency', 'f (THz)', 1 / PRE.T)
    TIME = Parameter('Time', 't (ps)', 1 / PRE.p)

    PHASE = Parameter('Phase Profile', 'φ (rad)', 1)
    AMPLITUDE = Parameter('Amplitude Profile', 'A (V/m)', 1)
    FIELD = Parameter('Real Field Components', 'E (V/m)', 1)
    INTENSITY = Parameter('Intensity Profile', 'I (GW/m²)', 1 / PRE.G)
    WIGNER = Parameter('Wigner Distribution', '', 1)
