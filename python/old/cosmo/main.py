""" Sample script to run and plot two-point correlation function using
module correlation_function """

import numpy
import matplotlib
import matplotlib.pyplot
from correlation_function import CorrelationFunction
from cosmology import Cosmology

matplotlib.rc('font', size=15)


def main():
    """ Run and plot two-point correlation function using
    class correlation_function.CorrelationFunction """

    # First define and set cosmological parameters
    # H0=Hubble constant at z=0
    # Om0=Omega matter at z=0
    # Ob0=Omega baryonic matter at z=0
    # Ode0=Omega dark energy at z=0
    cosmo = Cosmology()
    cosmo.set_model(H0=100, Om0=0.307, Ob0=0.0486, Ode0=0.693)

    # Constructor takes in configuration file names
    tpcf = CorrelationFunction('sample_config.cfg')

    # Get normalization factor (both weighted and unweighted)
    norm = numpy.array([tpcf.normalization(weighted=True),
                        tpcf.normalization(weighted=False)])

    # Construct separation distribution RR(s) between pairs of randoms
    rand_rand, err_rand_rand, bins_s = tpcf.rand_rand(cosmo)
    rand_rand[0] /= norm[0][0]  # normalize weighted RR(s)
    rand_rand[1] /= norm[1][0]  # normalize unweighted RR(s)

    # Construct separation distribution DR(s) between pairs of a random point
    # and a galaxies
    data_rand, err_data_rand, _ = tpcf.data_rand(cosmo)
    data_rand[0] /= norm[0][1]  # normalize weighted DR(s)
    data_rand[1] /= norm[1][1]  # normalize unweighted DR(s)

    # Construct separation distribution DD(s) between pairs of galaxies
    data_data, err_data_data, _ = tpcf.data_data(cosmo)
    data_data[0] /= norm[0][2]  # normalize weighted DD(s)
    data_data[1] /= norm[1][2]  # normalize unweighted DD(s)

    # Construct two-point correlation function, both weighted and unweighted
    correlation = numpy.zeros((2, 2, bins_s.size-1))
    correlation[0] = tpcf.correlation(rand_rand[0], data_rand[0], data_data[0],
                                      bins_s)
    correlation[1] = tpcf.correlation(rand_rand[1], data_rand[1], data_data[1],
                                      bins_s)

    # Save RR(s), DR(s), DD(s) and tpcf into .npz format
    numpy.savez("out/tpcf_sample",
                RR=rand_rand, DR=data_rand, DD=data_data,
                TPCF=correlation[0], TPCFSS=correlation[1],
                BINS=bins_s)

    # Create plot figure for RR(s), DR(s), DD(s) and tpcf
    figure, axes = matplotlib.pyplot.subplots(2, 2, figsize=(12, 8), sharex=True)

    bins_center = 0.5*(bins_s[:-1]+bins_s[1:])
    # Plot weighted RR(s), DR(s), DD(s)
    axes[0, 0].plot(bins_center, rand_rand[0], label='RR(s)')
    axes[0, 0].plot(bins_center, data_rand[0], label='DR(s)')
    axes[0, 0].plot(bins_center, data_data[0], label='DD(s)')
    # Plot unweighted RR(s), DR(s), DD(s)
    axes[0, 1].plot(bins_center, rand_rand[1], label='RR(s)')
    axes[0, 1].plot(bins_center, data_rand[1], label='DR(s)')
    axes[0, 1].plot(bins_center, data_data[1], label='DD(s)')
    # Plot weighted and unweighted tpcf
    axes[1, 0].plot(bins_center, correlation[0][0], label='Weighted')
    axes[1, 0].plot(bins_center, correlation[1][0], label='Unweighted')
    # Plot weighted and unweighted tpcfs^2
    axes[1, 1].plot(bins_center, correlation[0][1], label='Weighted')
    axes[1, 1].plot(bins_center, correlation[1][1], label='Unweighted')

    # Set axis labels
    axes[0, 0].set(ylabel='Normed Counts', title='Weighted')
    axes[0, 1].set(ylabel='Normed Counts', title='Unweighted')
    axes[1, 0].set(ylabel=r'$\xi(s)$')
    axes[1, 1].set(ylabel=r'$\xi(s)s^{2}$')
    for ax in axes.flat:
        ax.set(xlabel='s [Mpc/h]')
        ax.legend()

    figure.tight_layout()
    # Save figure
    matplotlib.pyplot.savefig("plot/sample.png", bbox_inches='tight')


if __name__ == "__main__":
    main()
