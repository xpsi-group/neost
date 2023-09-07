from __future__ import division

__all__ = ["_verbose",
           "_rank",
           "_size",
           "_pi",
           "_2pi",
           "_4pi",
           "_c",
           "_csq",
           "_G",
           "_M_s",
           "_km"]

import math as _m

# try:
#     _sys
# except NameError:
#     import sys as _sys

_pi = _m.pi
_2pi = 2.0 * _pi
_4pi = 4.0 * _pi
_c = 2.99792458E10  # cgs
_csq = _c * _c  # cgs
_kpc = 3.08567758e19  # SI
_keV = 1.60217662e-16  # SI
_G = 6.6730831e-8  # cgs
_M_s = 1.9887724767047002e33  # cgs
_h = 6.62607004e-34  # SI
_dpr = 180.0 / _pi
_km = 1.0e5  # cgs
_rhons = 267994004080000.03  # cgs
_dyncm2_to_MeVfm3 = 1. / (1.6022e33)
_gcm3_to_MeVfm3 = 1. / (1.7827e12)
_oneoverfm_MeV = 197.33
_n_ns = 0.16

try:
    from mpi4py import MPI

except ImportError:
    _rank = 0
    _verbose = False
else:
    _rank = MPI.COMM_WORLD.rank
    _size = MPI.COMM_WORLD.size
    if _rank == 0:
        _verbose = True
    else:
        _verbose = False


class NEOSTError(Exception):
    """ Base exception for NEOST-specific runtime errors. """
