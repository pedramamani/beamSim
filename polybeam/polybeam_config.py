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
    transform_nonplanar = 'Transform failed since some of the cross-sections are not planar.'
    plot_nonplanar = 'Cannot plot since some of the cross-sections are not planar.'
    disperse = 'Some frequency components don\'t have a first-order diffraction for these grating parameters.'
    chirp_alias = 'Chirp causes phase aliasing. Make any of the following changes to relieve:\n' \
                  'α < {:.0f}ps² or Nf >= 2^{:.0f} or ηf < {:.1f}.'



class WRAP:
    @dataclass
    class Method:
        type: int  # 0 for initializer, 1 for modifier, and 2 for getter
        attribute: str
        message: str
        
    class Type:
        initializer = 0
        modifier = 1
        getter = 2

    __init__ = Method(Type.initializer, '', 'Initialize f0={f0:.0f}THz, Δf={Δf:.1f}THz, Δx={Δx:.1f}mm')
    rotate = Method(Type.modifier, '', 'Rotate α={α:.1f}°')
    mask = Method(Type.modifier, '', 'Apply mask')
    propagate = Method(Type.modifier, '', 'Propagate Δz={Δz:.1f}cm')
    lens = Method(Type.modifier, '', 'Apply lens f={f:.1f}cm')
    disperse = Method(Type.modifier, '', 'Disperse d={d:d}mm⁻¹, α={α:.1f}°')
    chirp = Method(Type.modifier, '', 'Add chirp α={α:.1f}ps²')

    get_field_time = Method(Type.getter, 'Et', 'Get temporal field')
    get_phase_time = Method(Type.getter, 'φt', 'Get temporal phase')
    get_amplitude_time = Method(Type.getter, 'At', 'Get temporal amplitude')
    get_intensity_time = Method(Type.getter, 'It', 'Get temporal intensity')
    get_field_frequency = Method(Type.getter, 'Ef', 'Get spectral field')
    get_phase_frequency = Method(Type.getter, 'φf', 'Get spectral phase')
    get_amplitude_frequency = Method(Type.getter, 'Af', 'Get spectral amplitude')
    get_intensity_frequency = Method(Type.getter, 'If', 'Get spectral intensity')
    get_wigner = Method(Type.getter, 'W', 'Get Wigner distribution')

    time = '[{:.2f}s].'


class PLOT:
    @dataclass
    class Variable:
        title: str
        label: str
        scale: float
        attribute: str

    position = Variable('Position', 'x (mm)', 1 / PRE.m, 'xs')
    frequency = Variable('Frequency', 'f (THz)', 1 / PRE.T, 'fs')
    time = Variable('Time', 't (ps)', 1 / PRE.p, 'ts')
    phase = Variable('Phase Profile', 'φ (rad)', 1, 'φs')
    amplitude = Variable('Amplitude Profile', 'A (V/m)', 1, 'As')
    field = Variable('Field Values', 'E (V/m)', 1, 'Es')
    intensity = Variable('Intensity Profile', 'I (GW/m²)', 1 / PRE.G, 'Is')
    wigner = Variable('Wigner Distribution', '', 1, 'W')
