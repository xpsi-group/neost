import numpy as np


def m1_from_mc_m2(mc, m2):

    num1 = (2. / 3.)**(1. / 3.) * mc**5.
    denom1 = ((9 * m2**7. * mc**5. + np.sqrt(3.) * 
              np.sqrt(abs(27 * m2**14. * mc**10. - 
                         4. * m2**9. * mc**15.)))**(1. / 3.))
    denom2 = 2.**(1. / 3.) * 3.**(2. / 3.) * m2**3.
    return num1 / denom1 + denom1 / denom2


def m1_m2_from_mc_q(chirp, q):
    m1 = chirp * (1. + q)**(1. / 5.) * q**(-3. / 5.)
    m2 = chirp * (1. + q)**(1. / 5.) * q**(2. / 5.)
    return m1, m2


def tidaltilde(m1, m2, l1, l2):
    tidal = (16. / 13. * ((m1 + 12. * m2) * m1**4. * l1 + (m2 + 12. * m1) * 
             m2**4. * l2) / (m1 + m2)**5.)
    return tidal
