from pathlib import Path
from os import path

c = 2.9979E8
i = complex(0, 1)
π = 3.1416
μ0 = 1.2566E-6

fx = 32
fR = 2  # todo: maybe should be set dynamically
δz = 1E-4
DEFAULT_Nx = 2 ** 16
DEFAULT_I0 = 1E9

ASSETS_DIR = Path(path.dirname(__file__)) / 'monobeam_assets'
WISDOM_FILE = 'pyfftw_wisdom_{}'


class ERROR:
    ROTATION = 'Rotation results in phase aliasing. Decrease rotation angle or increase sampling to 2^{:d}.'


class PROPAGATOR:
    IN2IN = (True, True)
    IN2OUT = (True, False)
    OUT2IN = (False, True)
    OUT2OUT = (False, False)


class PLOT:
    POSITION_LABEL = 'x (mm)'
    POSITION_SCALE = 1E3

    PHASE_TITLE = 'Phase Profile'
    PHASE_LABEL = 'φ (rad)'

    AMPLITUDE_TITLE = 'Amplitude Profile'
    AMPLITUDE_LABEL = 'A (V/m)'

    INTENSITY_TITLE = 'Intensity Profile'
    INTENSITY_LABEL = 'I (W/m²)'

    LOG_PLANAR = 'Plotting "{}" on planar cross-section.'
    LOG_SPHERICAL = 'Plotting "{}" on spherical cross-section with curvature {:.2f}m.'



