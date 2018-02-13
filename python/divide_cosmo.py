""" Script for submitting jobs to calculate DD(s), DR(s), and RR(s) """

# Python modules
import sys
import configparser
import pickle

# User-defined module
from lib.cosmolib.catalog import DataCatalog, DistrCatalog
from lib.cosmolib.helper import Bins, JobHelper, CorrelationHelper


def initialize_catalog(config_fname):
    """ Initialize catalog and binning """
    # Read from configuration file
    config = configparser.SafeConfigParser()
    config.read(config_fname)

    # Set binnings
    bins = Bins(config['LIMIT'], config['BINWIDTH'])

    # Initialize catalog
    rand_type = config['RANDOM']['type'].lower()
    if rand_type == 'data_catalog':
        # Import randoms catalog as FITS catalog
        rand_catalog = DataCatalog(config['RANDOM'], bins.limit)
        rand_catalog = rand_catalog.to_distr(bins.limit, bins.num_bins)

    elif rand_type == 'distr_catalog':
        # Import randoms catalog as NPZ distribution
        rand_catalog = DistrCatalog(config['RANDOM'])

    elif rand_type == 'pickle':
        # Import randoms catalog as pickles
        pickle_in = open(config['RANDOM']['path'], 'rb')

        # Comparing two binning scheme
        test_bins = pickle.load(pickle_in)
        if not bins == test_bins:
            raise RuntimeError("Bins do not match.")

        print("Import catalog from Pickle: {}".format(config['RANDOM']['path']))
        rand_catalog = pickle.load(pickle_in)
        pickle_in.close()

    else:
        raise ValueError('TYPE must be "data_catalog", "distr_catalog", or "pickle"')
    data_catalog = DataCatalog(config['GALAXY'], bins.limit)

    return data_catalog, rand_catalog, bins


def main():
    """ Main
    Cmd arguments are job number, total number of jobs, configuration file,
    and output prefix
      + Job number: Job number must be an integer from 0 to total_job-1.
      + Total jobs: Total number of jobs that will be submitted.
      + Configuration: Setting for correlation function. See below.
      + Prefix: Prefix of the output files (can include folder path, folder must
      exist).
    If job number is 0, will also save comoving distribution P(r) and
    normalization factor."""
    print("DIVIDE module")

    # Read in cmd argument
    no_job = int(sys.argv[1])
    total_jobs = int(sys.argv[2])
    config_fname = sys.argv[3]
    prefix = sys.argv[4]

    # Set job helper
    job_helper = JobHelper(total_jobs)
    job_helper.set_current_job(no_job)

    # Initialize catalog and binnings
    data, rand, bins = initialize_catalog(config_fname)

    # Calculate f(theta), and g(theta, r)
    theta_distr, _ = rand.theta_distr(bins.max('theta'), bins.nbins('theta'), job_helper)
    z_theta_distr, _, _ = rand.z_theta_distr(
        data, bins.max('theta'), bins.nbins('theta'), job_helper)

    # Initialize save object
    save_object = CorrelationHelper(total_jobs)
    save_object.set_theta_distr(theta_distr)
    save_object.set_r_theta_distr(z_theta_distr)
    if no_job == 0:
        save_object.set_r_distr(rand.z_distr)
        save_object.set_bins(bins)
        save_object.set_norm(data.norm(), rand.norm(data), rand.norm())

    pickle_out = open("{}_{:03d}.pkl".format(prefix, no_job), "wb")
    pickle.dump(save_object, pickle_out, protocol=-1)
    if no_job == 0:
        pickle.dump(data, pickle_out, protocol=-1) # save data catalog
    pickle_out.close()


if __name__ == "__main__":
    main()
