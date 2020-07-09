from pathlib import Path
from os import path

c = 2.9979E8
μ0 = 1.2566E-6

fν = 32
fx = 8
DEFAULT_Nν = 2 ** 8
DEFAULT_Nx = 2 ** 15
DEFAULT_I0 = 1E9

ASSETS_DIR = Path(path.dirname(__file__)) / 'polybeam_assets'
WISDOM_FILE = 'pyfftw_wisdom_{}'


class ERROR:
    TRANSFORM = 'Transform failed since one or more of the wavefronts are not planar.'
    DISPERSION = 'Some frequency components don\'t have a first-order diffraction for these grating parameters.'


class PLOT:
    POSITION_LABEL = 'x (mm)'
    POSITION_SCALE = 1E3

    FREQUENCY_LABEL = 'ν (THz)'
    FREQUENCY_SCALE = 1E-12

    TIME_LABEL = 't (fs)'
    TIME_SCALE = 1E15

    PHASE_TITLE = 'Phase Profile'
    PHASE_LABEL = 'φ (rad)'

    AMPLITUDE_TITLE = 'Amplitude Profile'
    AMPLITUDE_LABEL = 'A (V/m)'
    AMPLITUDE_SCALE = 1E0

    INTENSITY_TITLE = 'Intensity Profile'
    INTENSITY_LABEL = 'I (MW/m²)'
    INTENSITY_SCALE = 1E-6
