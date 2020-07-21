from pathlib import Path
from os import path

i = complex(0, 1)
c = 2.9979E8
π = 3.14159
μ0 = 1.2566E-6

ηx = 32
ηR = 2  # todo: should be set dynamically?
δz = 1E-4
DEFAULT_Nx = 2 ** 16
DEFAULT_I0 = 1  # (TW/m²)

ASSETS_DIR = Path(path.dirname(__file__)) / 'monobeam_assets'
WISDOM_FILE = 'pyfftw_wisdom_{}'


class PRE:
    c = 1E-2
    m = 1E-3
    T = 1E12
    G = 1E9


class ERROR:
    ROTATION = 'Rotation introduces phase aliasing. Decrease rotation angle or increase sampling to 2^{:.0f}.'


class PROPAGATOR:
    IN2IN = (True, True)
    IN2OUT = (True, False)
    OUT2IN = (False, True)
    OUT2OUT = (False, False)


class PLOT:
    POSITION_LABEL = 'x (mm)'
    POSITION_SCALE = 1 / PRE.m

    PHASE_TITLE = 'Phase Profile'
    PHASE_LABEL = 'φ (rad)'

    AMPLITUDE_TITLE = 'Amplitude Profile'
    AMPLITUDE_LABEL = 'A (V/m)'
    AMPLITUDE_SCALE = 1

    INTENSITY_TITLE = 'Intensity Profile'
    INTENSITY_LABEL = 'I (GW/m²)'
    INTENSITY_SCALE = 1 / PRE.G

    LOG_PLANAR = 'Plotting "{}" on planar cross-section.'
    LOG_SPHERICAL = 'Plotting "{}" on spherical cross-section with curvature {:.1f}cm.'



