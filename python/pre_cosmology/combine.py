""" Script for combining job results and calculate DD(s), DR(s), and RR(s) """

import os
import sys
import glob
import numpy
from correlation_function import CorrelationFunction

def main():
    """ Main """
    # Cmd argument is output prefix
    prefix = sys.argv[1]  # prefix can include directory name

    # Combining histogram by simply taking the sum in each bin
    fname_list = sorted(glob.glob("{}*.npz".format(prefix)))
    for i, fname in enumerate(fname_list):
        if fname == "{}_final.npz".format(prefix):
            continue
        print("Reading from {}".format(fname))
        temp_file = numpy.load(fname)
        if i is 0:
            data_data = temp_file["DD"]
            theta_hist = temp_file["ANGULAR_D"]
            theta_r_hist = temp_file["ANGULAR_R"]
            r_hist = temp_file["R_HIST"]
            norm = temp_file["NORM"]
        else:
            data_data += temp_file["DD"]
            theta_hist += temp_file["ANGULAR_D"]
            theta_r_hist += temp_file["ANGULAR_R"]

    # Create an instance of two-point correlation function that reads in
    # configuration file
    config_fname = "{}_config.cfg".format(prefix)
    if not os.path.isfile(config_fname):
        raise IOError("Configuration file not found.")
    tpcf = CorrelationFunction(config_fname, import_catalog=True)

    # Calculate RR(s) and DR(s), DD(s)
    rand_rand, bins_s = tpcf.rand_rand(theta_hist, r_hist)
    data_rand, _ = tpcf.data_rand(theta_r_hist, r_hist)
    # Get error
    err_rand_rand = tpcf.get_error(rand_rand[0], rand_rand[1])
    err_data_rand = tpcf.get_error(data_rand[0], data_rand[1])
    err_data_data = tpcf.get_error(data_data[0], data_data[0])
    # Normalize
    for i in range(2):
        rand_rand[i] = rand_rand[i]/norm[i][0]
        data_rand[i] = data_rand[i]/norm[i][1]
        data_data[i] = data_data[i]/norm[i][2]
        err_rand_rand[i] /= numpy.sqrt(norm[i][0])
        err_data_rand[i] /= numpy.sqrt(norm[i][1])
        err_data_data[i] /= numpy.sqrt(norm[i][2])

    # Construct two-point correlation function, both weighted and unweighted
    correlation = numpy.zeros((2, 2, bins_s.size-1))
    correlation[0] = tpcf.correlation(rand_rand[0], data_rand[0], data_data[0],
                                      bins_s)
    correlation[1] = tpcf.correlation(rand_rand[1], data_rand[1], data_data[1],
                                      bins_s)

    # Save results
    numpy.savez("{}_final".format(prefix),
                DD=data_data, RR=rand_rand, DR=data_rand,
                ERR_DD=err_data_data, ERR_RR=err_rand_rand, ERR_DR=err_data_rand,
                TPCF=correlation[:, 0], TPCFSS=correlation[:, 1],
                BINS=bins_s)


if __name__ == "__main__":
    main()
