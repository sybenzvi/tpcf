""" Script to generat unittest data """

import numpy
from correlation_function import CorrelationFunction
from cosmology import Cosmology

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

    # Generate data for unittest
    tpcf = CorrelationFunction('unittest/config_test.cfg')
    norm = numpy.array([tpcf.normalization(weighted=True),
                        tpcf.normalization(weighted=False)])
    rand_rand, err_rand_rand, bins_s = tpcf.rand_rand(cosmo)
    data_rand, err_data_rand, _ = tpcf.data_rand(cosmo)
    data_data, err_data_data, _ = tpcf.data_data(cosmo)

    z_hist = tpcf.get_redshift_hist()
    theta_hist = tpcf.get_angular_distance_hist()
    theta_z_hist = tpcf.get_angular_redshift_hist()

    # Construct two-point correlation function, both weighted and unweighted
    correlation_w = tpcf.correlation(rand_rand[0], data_rand[0], data_data[0],
                                     bins_s)
    correlation_u = tpcf.correlation(rand_rand[1], data_rand[1], data_data[1],
                                     bins_s)

    # Save RR(s), DR(s), DD(s) and tpcf into .npz format
    numpy.savez("unittest/tpcf_test",
                RR=rand_rand, DR=data_rand, DD=data_data,
                E_RR = err_rand_rand, E_DR=err_data_rand, E_DD=err_data_data,
                TPCF_W=correlation_w, TPCF_U=correlation_u,
                Z_HIST=z_hist, ANGULAR_D_HIST=theta_hist, ANGULAR_Z_HIST=theta_z_hist,
                BINS=bins_s)


if __name__ == "__main__":
    main()
