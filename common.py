import contextlib
from os import path
import pyfftw
import pickle


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
