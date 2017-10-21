""" Generate a flat distribution for unittest data """

import sys
import numpy
from astropy.io import fits

DEG2RAD = numpy.pi/180.
RAD2DEG = 180./numpy.pi

def main():
    """ Main """
    ndata = int(sys.argv[1])
    out_fname = sys.argv[2]

    # generate flat data
    ra_min = 110.
    ra_max = 260.
    dec_min = -4.
    dec_max = 55
    z_min = 0.43
    z_max = 0.70

    sindec_min = numpy.sin(DEG2RAD*dec_min)
    sindec_max = numpy.sin(DEG2RAD*dec_max)

    # uniformly generate on a sphere
    ra = numpy.random.random(ndata)*(ra_max-ra_min)+ra_min
    sindec = numpy.random.random(ndata)*(sindec_max-sindec_min)+sindec_min
    dec = RAD2DEG*numpy.arcsin(sindec)
    # generate random redshift
    z = numpy.random.random(ndata)*(z_max-z_min)+z_min
    weight = numpy.ones_like(z)

    # save data into .fits format
    c1 = fits.Column(name="ra", array=ra, format="D")
    c2 = fits.Column(name="dec", array=dec, format="D")
    c3 = fits.Column(name="z", array=z, format="D")
    c4 = fits.Column(name="weight", array=weight, format="D")
    t = fits.BinTableHDU.from_columns([c1, c2, c3, c4])
    t.writeto(out_fname)

if __name__ == "__main__":
    main()