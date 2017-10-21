""" Script for submitting jobs to calculate DD(s), DR(s), and RR(s) """

import sys
import subprocess
import numpy
from correlation_function import CorrelationFunction

def main():
    """ Main """
    # Cmd arguments are job number, total number of jobs, configuration file,
    # and output prefix
    #   + Job number: Job number must be an integer from 0 to total_job-1.
    #   + Total jobs: Total number of jobs that will be submitted.
    #   + Configuration: Setting for correlation function. See below.
    #   + Prefix: Prefix of the output files (can include folder path, folder must
    #   exist).
    # If job number is 0, will also save comoving distribution P(r) and
    # normalization factor and cache configuration file.
    no_job = int(sys.argv[1])
    total_jobs = int(sys.argv[2])
    config_fname = sys.argv[3]
    prefix = sys.argv[4]

    # Calculate child-process data
    print("Job number: {}. Total jobs: {}.".format(no_job, total_jobs))
    # Save configuration files
    if no_job is 0:
        subprocess.call("cp {} {}_config.cfg".format(config_fname, prefix).split())

    # Create an instance of two-point correlation function that reads in
    # configuration file
    tpcf = CorrelationFunction(config_fname)
    # Angular distance distribution f(theta)
    theta_hist, _ = tpcf.angular_distance(no_job, total_jobs)
    # Radial angular distribution g(theta, r)
    theta_z_hist, _, _ = tpcf.angular_redshift(no_job, total_jobs)

    # Save with prefix
    if no_job is 0:
        # Save comoving distribution P(r) and normalization constant
        z_hist, _ = tpcf.redshift_distribution()
        norm = numpy.array([tpcf.normalization(weighted=True),
                            tpcf.normalization(weighted=False)])
        numpy.savez("{}_{:03d}".format(prefix, no_job),
                    ANGULAR_D=theta_hist, ANGULAR_Z=theta_z_hist,
                    Z_HIST=z_hist, NORM=norm)
    else:
        numpy.savez("{}_{:03d}".format(prefix, no_job),
                    ANGULAR_D=theta_hist, ANGULAR_Z=theta_z_hist)


if __name__ == "__main__":
    main()
