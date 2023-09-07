import pytest
from neost.eos.base import BaseEoS
import neost.global_imports as global_imports

class TestBaseEoSInit(object):

    def setup_class(cls):
        cls.rho_ns = global_imports._rhons

    def test_BaseEoS_class_initializes(self):
        eos = BaseEoS()

    def test_BaseEoS_class_breaks_with_too_high_rhot(self):
        with pytest.raises(ValueError):
            eos = BaseEoS(rho_t=2.1*self.rho_ns)

    def test_BaseEoS_class_breaks_with_too_small_rhot(self):
        with pytest.raises(ValueError):
            eos = BaseEoS(rho_t=0.0)
