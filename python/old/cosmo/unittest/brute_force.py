""" Brute force calculation of tpcf """

import numpy
import matplotlib
import matplotlib.pyplot
from astropy.io import fits

def import_fits(fname):
    """ Simple function to import fits, convert to cartesian catalog """
    hdulist = fits.open(fname)
    tbdata = hdulist[1].data
    temp_ra = tbdata["ra"]
    temp_dec = tbdata["dec"]
    temp_r = tbdata["r"]
    catalog = numpy.array([temp_r*numpy.cos(temp_dec)*numpy.cos(temp_ra),
                           temp_r*numpy.cos(temp_dec)*numpy.sin(temp_ra),
                           temp_r*numpy.sin(temp_dec)]).T
    hdulist.close()
    return catalog

def get_distance(x, y):
    """ Get distance to point x to point y using cartesian coordinates """
    return numpy.sqrt(numpy.sum((x-y)**2, axis=1))

def main():
    """ Main """
    rand_cat = import_fits("randoms_unittest.fits")
    data_cat = import_fits("galaxies_unittest.fits")

    # normalization
    n_rand = rand_cat.shape[0]
    n_data = data_cat.shape[0]
    norm_rand_rand = 0.5*n_rand*(n_rand-1)
    norm_data_rand = n_rand*n_data
    norm_data_data = 0.5*n_data*(n_data-1)

    bins_s = numpy.linspace(0., 200., 51)

    # RR
    print("Calculate RR(s)")
    rand_rand = numpy.zeros(50)
    for i, point in enumerate(rand_cat):
        d = get_distance(point, rand_cat[i+1:])
        hist, _ = numpy.histogram(d, bins=50, range=(0, 200))
        rand_rand += hist
    rand_rand = 1.*rand_rand/norm_rand_rand

    # DR
    print("Calculate DR(s)")
    data_rand = numpy.zeros(50)
    for i, point in enumerate(data_cat):
        d = get_distance(point, rand_cat)
        hist, _ = numpy.histogram(d, bins=50, range=(0, 200))
        data_rand += hist
    data_rand = 1.*data_rand/norm_data_rand

    # DD
    print("Calculate DD(s)")
    data_data = numpy.zeros(50)
    for i, point in enumerate(data_cat):
        d = get_distance(point, data_cat[i+1:])
        hist, _ = numpy.histogram(d, bins=50, range=(0, 200))
        data_data += hist
    data_data = 1.*data_data/norm_data_data

    # Correlation function
    correlation = data_data-2*data_rand+rand_rand
    correlation = numpy.divide(correlation, rand_rand,
                               out= numpy.zeros_like(data_data),
                               where=rand_rand!=0)
    correlation_ss = 0.5*(bins_s[:-1]+bins_s[1:])*correlation

    # Create plot figure for RR(s), DR(s), DD(s) and tpcf
    figure, axes = matplotlib.pyplot.subplots(1, 3, figsize=(12, 5), sharex=True)

    bins_center = 0.5*(bins_s[:-1]+bins_s[1:])
    # Plot weighted RR(s), DR(s), DD(s)
    axes[0].plot(bins_center, rand_rand, label='RR(s)')
    axes[0].plot(bins_center, data_rand, label='DR(s)')
    axes[0].plot(bins_center, data_data, label='DD(s)')
    # Plot weighted and unweighted tpcf
    axes[1].plot(bins_center, correlation)
    # Plot weighted and unweighted tpcfs^2
    axes[2].plot(bins_center, correlation_ss)

    # Set axis labels
    axes[0].set(ylabel='Normed Counts')
    axes[1].set(ylabel=r'$\xi(s)$')
    axes[2].set(ylabel=r'$\xi(s)s^{2}$')
    for axis in axes:
        axis.set(xlabel='s [Mpc/h]')
        axis.legend()

    figure.tight_layout()
    matplotlib.pyplot.show()


if __name__ == "__main__":
    main()