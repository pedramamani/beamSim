from pathlib import Path
from os import path
from dataclasses import dataclass

i = complex(0, 1)
c = 2.9979E8
π = 3.14159
μ0 = 1.2566E-6

ηR = 2  # todo: should be set dynamically?
DEFAULT_ηx = 16
DEFAULT_Nx = 2 ** 16
DEFAULT_I0 = 1  # (TW/m²)
δz = 1E-4

ASSETS_DIR = Path(path.dirname(__file__)) / 'monobeam_assets'
WISDOM_FILE = 'pyfftw_wisdom_{}'


class PRE:
    c = 1E-2
    m = 1E-3
    T = 1E12
    G = 1E9


class ERROR:
    rotation = 'Rotation introduces phase aliasing. Decrease rotation angle or increase position sampling to 2^{:.0f}.'


class PROPAGATOR:
    in_to_in = (True, True)
    in_to_out = (True, False)
    out_to_in = (False, True)
    out_to_out = (False, False)


class PLOT:
    @dataclass
    class Variable:
        title: str
        label: str
        scale: float

    position = Variable('', 'x (mm)', 1 / PRE.m)
    phase = Variable('Phase Profile', 'φ (rad)', 1)
    amplitude = Variable('Amplitude Profile', 'A (V/m)', 1)
    field = Variable('Field Values', 'E (V/m)', 1)
    intensity = Variable('Intensity Profile', 'I (GW/m²)', 1 / PRE.G)
    wigner = Variable('Wigner Distribution', '', 1)

    planar = 'Plotting "{}" on planar cross-section.'
    spherical = 'Plotting "{}" on spherical cross-section with curvature {:.1f}cm.'
