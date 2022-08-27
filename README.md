# Beam Simulation
This codebase offers the following beam propagation modules (the latter is built upon the former):
 * __MonoBeam__ to propagate a monochromatic Gaussian beam
 * __PolyBeam__ to propagate a polychromatic Gaussian beam
 
 Each module contains methods that simulate the effect of various optical components on the beam. An input beam is propagated by chaining a number of these methods together. Plotting methods are provided to visualize the beam in the temporal and spectral domains.
 
 ## Quick Start
Python 3.8 or above is required. Obtain and install it from [python.org](https://www.python.org/downloads/). Set up your virtual environment by following this [guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) or using a package manager of your choice. Then, install requirements by `pip install -r requirements.txt`.

Run the examples provided in `__main__.py`. See example code and documentation to help design your own propagation routine.
