""" Script for combining job results and calculate DD(s), DR(s), and RR(s) """

import sys
import glob
import numpy
import correlation_function

def main():
    """ Main """
    # Cmd argument is output prefix
    prefix = sys.argv[1]  # prefix can include directory name

    # Combining histogram by simply taking the sum in each bin
    fname_list = sorted(glob.glob("{}*.npz".format(prefix)))
    for i, fname in enumerate(fname_list):
        if fname == "{}_final.npz":
            continue
        print("Reading from {}".format(fname))
        temp_file = numpy.load(fname)
        if i is 0:
            rand_rand = temp_file["RR"]
            data_rand = temp_file["DR"]
            data_data = temp_file["DD"]
            bins_s = temp_file["BINS_S"]
            norm = temp_file["NORM"]
        else:
            rand_rand += temp_file["RR"]
            data_rand += temp_file["DR"]
            data_data += temp_file["DD"]

    # Get error
    err_rand_rand = correlation_function.get_error(rand_rand[0], rand_rand[1])
    err_data_rand = correlation_function.get_error(data_rand[0], data_rand[1])
    err_data_data = correlation_function.get_error(data_data[0], data_data[0])

    for i in range(2):
        rand_rand[i] = rand_rand[i]/norm[i][0]
        data_rand[i] = data_rand[i]/norm[i][1]
        data_data[i] = data_data[i]/norm[i][2]
        err_rand_rand[i] /= numpy.sqrt(norm[i][0])
        err_data_rand[i] /= numpy.sqrt(norm[i][1])
        err_data_data[i] /= numpy.sqrt(norm[i][2])

    # Construct two-point correlation function, both weighted and unweighted
    correlation = numpy.zeros((2, 2, bins_s.size-1))
    correlation[0] = correlation_function.correlation(
        rand_rand[0], data_rand[0], data_data[0], bins_s)
    correlation[1] = correlation_function.correlation(
        rand_rand[1], data_rand[1], data_data[1], bins_s)

    # Save results
    numpy.savez("{}_final".format(prefix),
                DD=data_data, RR=rand_rand, DR=data_rand,
                ERR_DD=err_data_data, ERR_RR=err_rand_rand, ERR_DR=err_data_rand,
                TPCF=correlation[:, 0], TPCFSS=correlation[:, 1],
                BINS=bins_s)


if __name__ == "__main__":
    main()
