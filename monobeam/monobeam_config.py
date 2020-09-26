from pathlib import Path
from os import path
from dataclasses import dataclass

i = complex(0, 1)
c = 2.9979E8
π = 3.14159
μ0 = 1.2566E-6
ε = 1E-5  # small threshold value to filter phases with small amplitude

ηR = 2  # todo: should be set dynamically?
DEFAULT_ηx = 16
DEFAULT_Nx = 2 ** 16
DEFAULT_I0 = 1  # (TW/m²)
δz = 1E-4

ASSETS_DIR = Path(path.dirname(__file__)) / 'monobeam_assets'
WISDOM_FILE = 'pyfftw_wisdom_{}'

FILTER_PHASE = False


class PRE:
    c = 1E-2
    m = 1E-3
    T = 1E12
    G = 1E9


class ERROR:
    ROTATE_ALIAS = 'Rotation causes phase aliasing. Make any of the following changes to relieve:\n' \
                      '|α| < {:.1f}° or Nx >= 2^{:.0f} or ηx < {:.1f}.'


class PROPAGATOR:
    IN_TO_IN = (True, True)
    IN_TO_OUT = (True, False)
    OUT_TO_IN = (False, True)
    OUT_TO_OUT = (False, False)


class PLOT:
    @dataclass
    class Variable:
        TITLE: str
        LABEL: str
        SCALE: float

    POSITION = Variable('', 'x (mm)', 1 / PRE.m)
    PHASE = Variable('Phase Profile', 'φ (rad)', 1)
    AMPLITUDE = Variable('Amplitude Profile', 'A (V/m)', 1)
    FIELD = Variable('Field Values', 'E (V/m)', 1)
    INTENSITY = Variable('Intensity Profile', 'I (GW/m²)', 1 / PRE.G)
    WIGNER = Variable('Wigner Distribution', '', 1)

    PLANAR_WARN = 'Plotting "{}" on planar cross-section.'
    SPHERICAL_WARN = 'Plotting "{}" on spherical cross-section with curvature {:.1f}cm.'
