import pytest
from neost.Star import Star

class TestStarInit(object):

    def setup_class(cls):
        cls.epscent = 1.0

    def test_star_class_initializes(self):
        st = Star(self.epscent)
        
    def test_star_class_breaks_with_arguments_missing(self):
        with pytest.raises(TypeError):
            st = Star()
            
class TestStarSolveStructure(object):

    def setup_class(cls):
        cls.epscent = 1.0
        
    def test_star_class_breaks_with_wrong_input_type(self):
        st = Star(self.epscent)
        eps=1.0
        pres=1.0
        with pytest.raises(TypeError):
            st.solve_structure(eps,pres)
