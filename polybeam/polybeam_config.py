from pathlib import Path
from os import path

i = complex(0, 1)
c = 2.9979E8
π = 3.14159
μ0 = 1.2566E-6

ην = 32
ηx = 16
ςE = 1E-6  # factor specifying low amplitude values to filter for phase profile
DEFAULT_Nν = 2 ** 8  # 2 ** 9
DEFAULT_Nx = 2 ** 12  # 2 ** 16
DEFAULT_I0 = 1  # (TW/m²)

ASSETS_DIR = Path(path.dirname(__file__)) / 'polybeam_assets'
WISDOM_FILE = 'pyfftw_wisdom_{}'


class PRE:
    c = 1E-2
    m = 1E-3
    f = 1E-15
    T = 1E12
    G = 1E9


class ERROR:
    TRANSFORM = 'Transform failed since some of the cross-sections are not planar.'
    SECTION = 'Cannot plot since some of the cross-sections are not planar.'
    DISPERSION = 'Some frequency components don\'t have a first-order diffraction for these grating parameters.'


class LOG:
    INIT = 'Initialize ν0={:.0f}THz, Δν={:.1f}THz, Δx={:.1f}mm'
    ROTATE = 'Rotate α={:.1f}°'
    MASK = 'Mask'
    PROPAGATE = 'Propagate Δz={:.1f}cm'
    LENS = 'Lens f={:.1f}cm'
    DISPERSE = 'Disperse d={:d}mm⁻¹, α={:.1f}°'
    CHIRP = 'Chirp α={:.1f}fs²'
    TRANSFORM = 'Transform'

    WAIT = '... '
    TIME = '[{:.2f}s].'


class PLOT:
    POSITION_LABEL = 'x (mm)'
    POSITION_SCALE = 1 / PRE.m

    FREQUENCY_LABEL = 'ν (THz)'
    FREQUENCY_SCALE = 1 / PRE.T

    TIME_LABEL = 't (fs)'
    TIME_SCALE = 1 / PRE.f

    PHASE_TITLE = 'Phase Profile'
    PHASE_LABEL = 'φ (rad)'

    AMPLITUDE_TITLE = 'Amplitude Profile'
    AMPLITUDE_LABEL = 'A (V/m)'
    AMPLITUDE_SCALE = 1

    INTENSITY_TITLE = 'Intensity Profile'
    INTENSITY_LABEL = 'I (GW/m²)'
    INTENSITY_SCALE = 1 / PRE.G

    WIGNER_TITLE = 'Wigner Distribution'

    @staticmethod
    def SAVE_TIME(title): return '-'.join(title.lower().split() + ['time'])

    @staticmethod
    def SAVE_FREQ(title): return '-'.join(title.lower().split() + ['freq'])

    @staticmethod
    def SAVE_WIGNER(title): return '-'.join(title.lower().split())
