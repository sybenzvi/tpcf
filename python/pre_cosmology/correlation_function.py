""" Modules to construct two-point correlation function using KD-Tree
Estimators used: (DD-2*DR+RR)/RR,
where D and R are data and randoms catalogs respectively """
__version__ = 1.30

import configparser
import numpy
from sklearn.neighbors import KDTree, BallTree
from astropy.io import fits
from cosmology import Cosmology

DEG2RAD = numpy.pi/180.
RAD2DEG = 180./numpy.pi


def import_fits(fname_key, fits_reader, region, cosmo):
    """ Import data into 2-d array
    Inputs:
    + fname_key: string
        Key for data filename in reader.
    + fits_reader: dict
        Must have attributes: "INDEX"=index of headers. "RA", "DEC", "Z",
        "WEIGHT"=corresponded variable names in header.
    + region: dict
        Region of galaxies to import. Must have attributes: "dec_max", "dec_min"
        , "ra_max", "ra_min", "z_min", "z_max"
    + cosmo: cosmology.Cosmology
        Cosmological parameters to convert redshift to comoving distance.
    Outputs:
    + catalog: ndarray or tuple of ndarrays
        Return catalog format in each row [DEC, RA, R, WEIGHT].
    """
    print("Importing from: {}".format(fits_reader[fname_key]))

    header_index = int(fits_reader["index"])
    hdulist = fits.open(fits_reader[fname_key])
    tbdata = hdulist[header_index].data
    temp_dec = DEG2RAD*tbdata[fits_reader["dec"]]
    temp_ra = DEG2RAD*tbdata[fits_reader["ra"]]
    temp_z = tbdata[fits_reader["z"]]
    temp_r = cosmo.z2r(temp_z)
    try:
        temp_weight_fkp = tbdata[fits_reader["weight_fkp"]]
        temp_weight_noz = tbdata[fits_reader["weight_noz"]]
        temp_weight_cp = tbdata[fits_reader["weight_cp"]]
        temp_weight_sdc = tbdata[fits_reader["weight_sdc"]]
        temp_weight = (temp_weight_sdc * temp_weight_fkp *
                       (temp_weight_noz + temp_weight_cp - 1))
    except KeyError:
        temp_weight = tbdata[fits_reader["weight"]]

    catalog = numpy.array([temp_dec, temp_ra, temp_r, temp_weight]).T
    hdulist.close()

    # cut by region
    dec_min = DEG2RAD*float(region["dec_min"])
    dec_max = DEG2RAD*float(region["dec_max"])
    ra_min = DEG2RAD*float(region["ra_min"])
    ra_max = DEG2RAD*float(region["ra_max"])
    r_min = cosmo.z2r(float(region["z_min"]))
    r_max = cosmo.z2r(float(region["z_max"]))
    cut = ((dec_min <= temp_dec) & (temp_dec <= dec_max)
           & (ra_min <= temp_ra) & (temp_ra <= ra_max)
           & (r_min <= temp_r) & (temp_r <= r_max))

    print("Data size: {}".format(numpy.sum(cut)))
    return catalog[cut]


def celestial2cart(declination, right_ascension, radius):
    """ Convert Celestial coordiante into Cartesian coordinate system
        Conversion equation is:
         -x = r*cos(dec)*cos(ra)
         -y = r*cos(dec)*sin(ra)
         -z = r*sin(dec)
        Inputs:
        + declination: array
            Values of DEC, declination, in radian.
        + right_ascension: array
            Values of RA, right ascension, in radian.
        + radius: array
            Values of the radius. Should have dimension of length.
        Outputs:
        + cart_x: array
            Values of X, same unit with radius.
        + cart_y: array
            Values of Y, same unit with radius.
        + cart_z: array
            Values of Z. same unit with radius.
        """
    cart_x = radius*numpy.cos(declination)*numpy.cos(right_ascension)
    cart_y = radius*numpy.cos(declination)*numpy.sin(right_ascension)
    cart_z = radius*numpy.sin(declination)
    return cart_x, cart_y, cart_z


def hist2point(hist, bins_x, bins_y, exclude_zeros=True):
    """ Convert 2d histogram into data points with weight. Take bincenter as
        value of points.
        Inputs:
        + hist: 2-d array
            Values of 2d histogram (length(bins_x)-1, length(bins_y-1)).
        + bins_x: array
            Binedges in x-axis.
        + bins_y: array
            Binedges in y-axis.
        + exclude_zeros: bool, default=True
            If True, return non-zero weight bins only.
        Outputs:
        + catalog: ndarray or tuple of ndarrays
            Array of data points with weights. Format in each row [X, Y, Weight]
        """
    center_x = 0.5*(bins_x[:-1]+bins_x[1:])
    center_y = 0.5*(bins_y[:-1]+bins_y[1:])
    grid_x, grid_y = numpy.meshgrid(center_x, center_y)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    hist = hist.T.flatten()
    catalog = numpy.array([grid_x, grid_y, hist]).T
    if exclude_zeros:
        return catalog[hist != 0]
    return catalog


def get_distance(radius1, radius2, theta):
    """ Given two points at radius1 and radius2 with angular separation
    theta (in rad), calulate distance between points"""
    return numpy.sqrt(radius1**2+radius2**2-2*radius1*radius2*numpy.cos(theta))


def get_bins(x_min, x_max, binwidth):
    """ Return the binnings given min, max and width. Apply roundings to
    binwidth such that (max-min)/binwidth is an integer. Also return the new
    binwidth."""
    nbins = int(numpy.ceil((x_max-x_min)/binwidth))
    if nbins <= 0:
        raise Exception("Max must be greater than Min.")
    bins = numpy.linspace(x_min, x_max, nbins+1)
    binw = bins[1]-bins[0]
    return bins, binw


def get_job_index(no_job, total_jobs, job_size):
    """ Calculate start and end index based on job number, total number of jobs,
    and the size of the jobs
    Inputs:
    + no_job: int
        Job number. Must be greater or equal to 0 and less than total_jobs
        (0 <= no_job < total_jobs).
    + total_jobs: int
        Total number of jobs. Must be a positive integer.
    + job_size: int
        Size of the jobs. Must be a positive integer.
    Outputs:
    + job_range: tuple
        Return range index of the job
    """
    # Calculate start and end index based on job number and total number of
    # jobs.
    job_index = numpy.floor(numpy.linspace(0, job_size, total_jobs+1))
    job_index = job_index.astype(int)
    job_range = (job_index[no_job], job_index[no_job+1])
    return job_range


class CorrelationFunction():
    """ Class to construct two-point correlation function """
    def __init__(self, config_fname, import_catalog=True):
        """ Constructor takes in configuration file and sets up binning
        variables """
        # Initialize variables
        self.data_cat = None
        self.rand_cat = None
        self.__bins_ra = None
        self.__bins_dec = None
        self.__bins_r = None
        self.__bins_s = None
        self.__bins_theta = None
        self.set_configuration(config_fname, import_catalog)

    def __angular_distance_thread(self, angular_points, arc_tree, start, end):
        """ Thread function to calculate angular distance distribution f(theta)
        as one-dimensional histogram.
        Inputs:
        + angular_points: ndarray
            Angular distribution R(ra, dec) breaks into data points with
            proper weights. Format of each row must be [DEC, RA, WEIGHT].
        + arc_tree: ball tree
            Ball tree fill with data in angular catalog. For more details,
            refer to sklearn.neighbors.BallTree
        + start: int
            Starting index of galaxies in galaxies catalog: index_start = start.
        + end: int
            Ending index of galaxies in galaxies catalog: index_end = end-1.
        Outputs:
        + theta_hist: ndarrays or tuples of ndarrays
            Return values of f(theta).
        """
        # Define angular distance distribution f(theta) as histogram
        nbins_theta = self.__bins_theta.size-1
        theta_max = self.__bins_theta.max()
        theta_hist = numpy.zeros(nbins_theta)

        print("Construct f(theta) from index {} to {}".format(start, end-1))
        for i, point in enumerate(angular_points[start:end]):
            if i % 10000 is 0:
                print(i)
            index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                                 r=theta_max,
                                                 return_distance=True)
            temp_weight = point[2]*angular_points[:, 2][index[0]]
            temp_hist, _ = numpy.histogram(theta[0], bins=nbins_theta,
                                           range=(0., theta_max),
                                           weights=temp_weight)
            theta_hist += temp_hist

        # Correction for double counting
        theta_hist = theta_hist/2.

        return theta_hist

    def __angular_comoving_thread(self, angular_points, arc_tree,
                                  start, end, mode):
        """ Thread function for calculating angular comoving distribution
        g(theta, r) as two-dimensional histogram.
        Inputs:
        + angular_points: ndarray
            Angular distribution R(ra, dec) breaks into data points with
            proper weights. Format of each row must be [DEC, RA, WEIGHT].
        + arc_tree: ball tree
            Ball tree fill with data in angular catalog. For more details,
            refer to sklearn.neighbors.BallTree
        + start: int
            Starting index of galaxies in galaxies catalog: index_start = start.
        + end: int
            Ending index of galaxies in galaxies catalog: index_end = end-1.
        + mode: string
            Must be either "data" or "angular". If "data", loop over galaxies
            catalog. If "angular", loop over angular distribution points.
        Outputs:
        + theta_r_hist: ndarray or tuple of ndarrays
            Return values of weighted and unweighted g(theta, r) respectively.
            Each has dimension (length(bins_theta)-1, length(bins_r)-1).
        """
        # Define angular-radial distribution g(theta, r) as weighted and
        # unweighted 2-d histogram respectively
        nbins_theta = self.__bins_theta.size-1
        nbins_r = self.__bins_r.size-1
        theta_max = self.__bins_theta.max()
        bins_range = ((0., theta_max),
                      (self.__bins_r.min(), self.__bins_r.max()))
        theta_r_hist = numpy.zeros((2, nbins_theta, nbins_r))

        print("Construct g(theta, r) from index {} to {}".format(start, end-1))
        if mode == "data":
            for i, point in enumerate(self.data_cat[start:end]):
                if i % 10000 is 0:
                    print(i)
                index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                                     r=theta_max,
                                                     return_distance=True)
                temp_r = numpy.repeat(point[2], index[0].size)
                # Fill unweighted histogram
                temp_weight = angular_points[:, 2][index[0]]
                temp_hist, _, _ = numpy.histogram2d(theta[0], temp_r,
                                                    bins=(nbins_theta, nbins_r),
                                                    range=bins_range,
                                                    weights=temp_weight)
                theta_r_hist[1] += temp_hist
                # Fill weighted histogram
                temp_weight = temp_weight*point[3]
                temp_hist, _, _ = numpy.histogram2d(theta[0], temp_r,
                                                    bins=(nbins_theta, nbins_r),
                                                    range=bins_range,
                                                    weights=temp_weight)
                theta_r_hist[0] += temp_hist
        elif mode == "angular":
            for i, point in enumerate(angular_points[start:end]):
                if i % 10000 is 0:
                    print(i)
                index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                                     r=theta_max,
                                                     return_distance=True)
                temp_r = self.data_cat[:, 2][index[0]]
                # Fill weighted histogram
                temp_weight = point[2]*self.data_cat[:, 3][index[0]]
                temp_hist, _, _ = numpy.histogram2d(theta[0], temp_r,
                                                    bins=(nbins_theta, nbins_r),
                                                    range=bins_range,
                                                    weights=temp_weight)
                theta_r_hist[0] += temp_hist
                # Fill unweighted histogram
                temp_weight = numpy.repeat(point[2], index[0].size)
                temp_hist, _, _ = numpy.histogram2d(theta[0], temp_r,
                                                    bins=(nbins_theta, nbins_r),
                                                    range=bins_range,
                                                    weights=temp_weight)
                theta_r_hist[1] += temp_hist

        return theta_r_hist

    def __pairs_separation_thread(self, point_cat, tree_cat, tree, start, end):
        """ Thread function for calculating separation distribution.
        Inputs:
        + point_cat: ndarray or tupless of ndarray
            Point catalog in Cartesian coordinates outside tree.
            Each row format [X,Y,Z].
        + tree_cat: ndarray or tupless of ndarray
            Galaxies catalog in Cartesian coordinates inside tree.
            Each row format [X,Y,Z].
        + tree: kd-tree
            K-D tree filled with points from tree_cat.
        + start: int
            Starting index of galaxies in galaxies catalog: index_start = start.
        + end: int
            Ending index of galaxies in galaxies catalog: index_end = end-1.
        Outputs:
        + pairs_seprtion: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted DD(s) respectively.
        """
        # Define weighted and unweighted DD(s) as two one-dimensional
        # histograms respectively.
        nbins_s = self.__bins_s.size-1
        s_max = self.__bins_s.max()
        pairs_separation = numpy.zeros((2, nbins_s))

        print("Calculate DD(s) from index {} to {}".format(start, end-1))
        for i, point in enumerate(point_cat[start:end]):
            if i % 10000 is 0:
                print(i)
            index, dist = tree.query_radius(point[: 3].reshape(1, -1),
                                            r=s_max,
                                            return_distance=True)
            # Fill weighted histogram
            temp_weight = tree_cat[:, 3][index[0]]*point[3]
            temp_hist, _ = numpy.histogram(dist[0], bins=nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            pairs_separation[0] += temp_hist
            # Fill unweighted histogram
            temp_hist, _ = numpy.histogram(dist[0], bins=nbins_s,
                                           range=(0., s_max))
            pairs_separation[1] += temp_hist

        # Correction for double counting
        if numpy.array_equal(point_cat, tree_cat):
            # sum of w^2
            pairs_separation[0][0] -= numpy.sum(point_cat[start:end, 3]**2)
            pairs_separation[1][0] -= end-start
            pairs_separation = pairs_separation/2.

        return pairs_separation

    def set_configuration(self, config_fname, import_catalog=True):
        """ Sets up binning variable based on input configuration file.
            If import_catalog is True, will import catalog data."""
        config = configparser.ConfigParser()
        config.read(config_fname)
        binnings = config['BINNING']
        region = config['REGION']

        # Create cosmology
        cosmo_params = config["COSMOLOGY"]
        cosmo = Cosmology()
        cosmo.set_model(float(cosmo_params["hubble0"]),
                        float(cosmo_params["omega_m0"]),
                        float(cosmo_params["omega_b0"]),
                        float(cosmo_params["omega_de0"]),
                        float(cosmo_params["temp_cmb"]),
                        float(cosmo_params["nu_eff"]),
                        list(map(float, cosmo_params["m_nu"].split(","))))

        # Import random and data catalogs
        if import_catalog:
            reader = config['FITS']
            self.data_cat = import_fits('data_filename', reader, region, cosmo)
            self.rand_cat = import_fits('random_filename', reader, region,
                                        cosmo)

        # Setting up binning variables
        # spatial separation (Mpc/h)
        s_max = float(region["s_max"])
        binwidth_s = float(binnings['binwidth_s'])
        self.__bins_s, binwidth_s = get_bins(0., s_max, binwidth_s)

        # comoving distance distribution(Mpc/h)
        r_min = cosmo.z2r(float(region["z_min"]))
        r_max = cosmo.z2r(float(region["z_max"]))
        if binnings['binwidth_r'] == 'auto':
            binwidth_r = binwidth_s/2.
        else:
            binwidth_r = float(binnings['binwidth_r'])
        self.__bins_r, binwidth_r = get_bins(r_min, r_max, binwidth_r)

        # declination, right ascension, and angular separation (rad)
        dec_min = DEG2RAD*float(region["dec_min"])
        dec_max = DEG2RAD*float(region["dec_max"])
        if binnings['binwidth_dec'] == 'auto':
            binwidth_dec = 1.*binwidth_r/r_max
        else:
            binwidth_dec = DEG2RAD*float(binnings['binwidth_dec'])
        self.__bins_dec, binwidth_dec = get_bins(dec_min, dec_max, binwidth_dec)

        ra_min = DEG2RAD*float(region["ra_min"])
        ra_max = DEG2RAD*float(region["ra_max"])
        if binnings['binwidth_ra'] == 'auto':
            binwidth_ra = 1.*binwidth_r/r_max
        else:
            binwidth_ra = DEG2RAD*float(binnings['binwidth_ra'])
        self.__bins_ra, binwidth_ra = get_bins(ra_min, ra_max, binwidth_ra)

        theta_max = DEG2RAD*float(region["theta_max"])
        if binnings['binwidth_theta'] == 'auto':
            binwidth_theta = 1.*binwidth_r/r_max
        else:
            binwidth_theta = DEG2RAD*float(binnings['binwidth_theta'])
        self.__bins_theta, binwidth_theta = get_bins(0., theta_max,
                                                     binwidth_theta)

        # Print out new configuration information
        print("Configuration:")
        print("Spatial separation (Mpc/h):")
        print("-[Min, Max, Binwidth] = [{}, {}, {}]".format(0., s_max,
                                                            binwidth_s))
        print("Comoving distribution (Mpc/h):")
        print("-[Min, Max, Binwidth] = [{}, {}, {}]".format(r_min, r_max,
                                                            binwidth_r))
        print("Declination (rad):")
        print("-[Min, Max, Binwidth] = [{}, {}, {}]".format(dec_min, dec_max,
                                                            binwidth_dec))
        print("Right ascension (rad):")
        print("-[Min, Max, Binwidth] = [{}, {}, {}]".format(ra_min, ra_max,
                                                            binwidth_ra))
        print("Angular separation (rad):")
        print("-[Min, Max, Binwidth] = [{}, {}, {}]".format(0., theta_max,
                                                            binwidth_theta))

    def comoving_distribution(self, catalog_type="random"):
        """ Calculate weighted and unweighted comoving distribution P(r) as
        two one-dimensional histograms.
        Inputs:
        + catalog_type: string (default=random)
            Catalog to calculate comoving distribution. Must be either
            "random" or "data".
        Outputs:
        + hist: array
            Values of weighted and unweighted P(r).
        + binedges: array
            Return binedges (length(hist)+1)
        """
        if self.data_cat is None or self.rand_cat is None:
            raise TypeError("Catalogs are not imported.")

        # Choose catalog based on input
        if catalog_type == "random":
            catalog = self.rand_cat
        else:
            catalog = self.data_cat

        # Calculate weighted and unweighted radial distribution P(r) as two
        # one-dimensional histograms respectively.
        r_hist = numpy.zeros((2, self.__bins_r.size-1))
        r_hist[0] += numpy.histogram(self.rand_cat[:, 2], bins=self.__bins_r,
                                     weights=self.rand_cat[:, 3])[0]
        r_hist[1] += numpy.histogram(self.rand_cat[:, 2], bins=self.__bins_r)[0]
        r_hist = 1.*r_hist/catalog.shape[0]

        return r_hist, self.__bins_r

    def angular_distance(self, no_job, total_jobs, leaf=40):
        """ Calculate the angular distance distribution f(theta) as an
        one-dimensional histogram. Binnings are defined in config file.
        Use a modified nearest-neighbors BallTree algorithm to calculate
        angular distance up to a given radius defined in config file.
        Inputs:
        + no_job: int
            Job number. Must be greater or equal to 0 and less than total_jobs
            (0 <= no_job < total_jobs).
        + total_jobs: int
            Total number of jobs. Must be a positive integer.
        + leaf: int (default=40)
            Number of points at which to switch to brute-force. For a specified
            leaf_size, a leaf node is guaranteed to satisfy
            leaf_size <= n_points <= 2*leaf_size, except in the case that
            n_samples < leaf_size. More details in sklearn.neighbors.BallTree.
        Outputs:
        + theta_hist: array
            Values of f(theta).
        + bins: array
            Binedges of histogram  (length(theta_hist)-1).
        """
        if self.data_cat is None or self.rand_cat is None:
            raise TypeError("Catalogs are not imported.")

        # Compute 2d angular distribution R(ra, dec) and breaks them into data
        # points with proper weights.
        angular_hist = numpy.histogram2d(self.rand_cat[:, 0],
                                         self.rand_cat[:, 1],
                                         bins=(self.__bins_dec, self.__bins_ra))
        angular_points = hist2point(*angular_hist)

        # Calculate start and end index based on job number and total number of
        # jobs.
        job_range = get_job_index(no_job, total_jobs, angular_points.shape[0])

        # Create a BallTree and use modified nearest-neighbors algorithm to
        # calculate angular distance up to a given radius.
        arc_tree = BallTree(angular_points[:, :2], leaf_size=leaf,
                            metric='haversine')
        theta_hist = self.__angular_distance_thread(angular_points, arc_tree,
                                                    *job_range)

        return theta_hist, self.__bins_theta

    def angular_comoving(self, no_job, total_jobs, leaf=40):
        """ Calculate g(theta, r), angular distance vs. comoving distribution,
        as a two-dimensional histogram. Binnings are defined in config file.
        Use a modified nearest-neighbors BallTree algorithm to calculate
        angular distance up to a given radius defined in config file.
        Inputs:
        + no_job: int
            Job number. Must be greater or equal to 0 and less than total_jobs
            (0 <= no_job < total_jobs).
        + total_jobs: int
            Total number of jobs. Must be a positive integer.
        + leaf: int (default=40)
            Number of points at which to switch to brute-force. For a specified
            leaf_size, a leaf node is guaranteed to satisfy
            leaf_size <= n_points <= 2*leaf_size, except in the case that
            n_samples < leaf_size. More details in sklearn.neighbors.BallTree.
        Outputs:
        + theta_r_hist: ndarray or tuple of ndarrays
            Return values of weighted and unweighted g(theta, r) respectively.
            Each has dimension (length(bins_theta)-1, length(bins_r)-1).
        + bins_theta: array
            Binedges along x-axis.
        + bins_r: array
            Binedges along y-axis.
        """
        if self.data_cat is None or self.rand_cat is None:
            raise TypeError("Catalogs are not imported.")

        # Compute 2d angular distribution R(ra, dec) and breaks them into data
        # points with proper weights.
        angular_hist = numpy.histogram2d(self.rand_cat[:, 0],
                                         self.rand_cat[:, 1],
                                         bins=(self.__bins_dec, self.__bins_ra))
        angular_points = hist2point(*angular_hist)

        # Temporary remove until further runtime analysis is done.
        # Optimizing: Runtime of BallTree modified nearest-neighbors is O(NlogM)
        # where M is the number of points in Tree and N is the number of points
        # for pairings. Thus, the BallTree is created using the size of the
        # smaller catalog, galaxies catalog vs. angular catalog.
        # if angular_points.shape[0] >= self.data_cat.shape[0]:
            # job_size = self.data_cat.shape[0]
            # mode = "data"
            # arc_tree = BallTree(angular_points[:, :2], leaf_size=leaf,
                                # metric='haversine')
        # else:
            # job_size = angular_points.shape[0]
            # mode = "angular"
            # arc_tree = BallTree(self.data_cat[:, :2], leaf_size=leaf,
                                # metric='haversine')
        # Temporarily only use angular points to fill Tree
        job_size = self.data_cat.shape[0]
        mode = "data"
        arc_tree = BallTree(angular_points[:, :2], leaf_size=leaf,
                            metric='haversine')

        # Calculate start and end index based on job number and total number of
        # jobs.
        job_range = get_job_index(no_job, total_jobs, job_size)
        theta_r_hist = self.__angular_comoving_thread(angular_points,
                                                      arc_tree,
                                                      job_range[0],
                                                      job_range[1], mode)

        return theta_r_hist, self.__bins_theta, self.__bins_r

    def rand_rand(self, theta_hist, r_hist):
        """ Calculate separation distribution RR(s) between pairs of randoms.
        Inputs:
        + theta_hist: array
            Values of angular distance distribution f(theta).
        + r_hist: ndarrays or tuples of ndarrays
            Values of the weighted and unweighted of the comoving distance
            distribution P(r) respective.
        Outputs:
        + rand_rand: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted RR(s) respectively.
        + bins: array
            Binedges of RR(s) (length(rand_rand_hist)+1).
        """
        # Convert f(theta) and P(r) into data points, weighted and unweighted
        weight = numpy.zeros((2, theta_hist.size, r_hist.shape[1]))
        for i in range(theta_hist.size):
            for j in range(r_hist.shape[1]):
                weight[0][i, j] = theta_hist[i]*r_hist[0][j]
                weight[1][i, j] = theta_hist[i]*r_hist[1][j]
        temp_points = [hist2point(weight[0], self.__bins_theta, self.__bins_r),
                       hist2point(weight[1], self.__bins_theta, self.__bins_r)]
        center_r = 0.5*(self.__bins_r[:-1]+self.__bins_r[1:])

        # Exclude zeros bins
        cut = r_hist[1] > 0
        center_r = center_r[cut]
        r_weight = numpy.zeros((2, center_r.size))
        r_weight[0] = r_hist[0][cut]
        r_weight[1] = r_hist[1][cut]

        # Define weighted and unweighted RR(s) as two one-dimensional
        # histograms respectively.
        nbins_s = self.__bins_s.size-1
        s_max = self.__bins_s.max()
        rand_rand = numpy.zeros((2, nbins_s))

        # Integration
        print("Construct RR(s)")
        for i, temp_r in enumerate(center_r):
            if i % 100 is 0:
                print(i)
            temp_s = get_distance(temp_r, temp_points[0][:, 1],
                                  temp_points[0][:, 0])

            # Fill weighted histogram
            temp_weight = r_weight[0][i]*temp_points[0][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, bins=nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            rand_rand[0] += temp_hist
            # Fill unweighted histogram
            temp_weight = r_weight[1][i]*temp_points[1][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, bins=nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            rand_rand[1] += temp_hist

        return rand_rand, self.__bins_s

    def data_rand(self, theta_r_hist, r_hist):
        """ Calculate separation distribution DR(s) between pairs of a random
        point and a galaxy.
        Inputs:
        + theta_r_hist: ndarrays or tuples of ndarrays
            Values of weighted and unweighted g(theta, r) respectively.
            Dimension must be (2, length(bins_theta)-1, length((bins_r)-1).
        + r_hist: ndarrays or tuples of ndarrays
            Values of the weighted and unweighted of the comoving distance
            distribution P(r) respective. Dimension must be
            (2, length(bins_r)-1).
        Outputs:
        + data_rand: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted DR(s) respectively.
        + bins: array
            Binedges of DR(s) (length(data_rand_hist)+1).
        """
        # Convert g(theta, r) into data points, weighted and unweighted
        temp_points = (
            (hist2point(theta_r_hist[0], self.__bins_theta, self.__bins_r),
             hist2point(theta_r_hist[1], self.__bins_theta, self.__bins_r)))
        center_r = 0.5*(self.__bins_r[:-1]+self.__bins_r[1:])

        # Exclude zeros bins
        cut = r_hist[1] > 0
        center_r = center_r[cut]
        r_weight = numpy.zeros((2, center_r.size))
        r_weight[0] = r_hist[0][cut]
        r_weight[1] = r_hist[1][cut]

        # Define weighted and unweighted DR(s) as two one-dimensional
        # histograms respectively.
        nbins_s = self.__bins_s.size-1
        s_max = self.__bins_s.max()
        data_rand = numpy.zeros((2, nbins_s))

        # Integration
        print("Construct DR(s)")
        for i, temp_r in enumerate(center_r):
            if i % 100 is 0:
                print(i)
            temp_s = get_distance(temp_r, temp_points[0][:, 1],
                                  temp_points[0][:, 0])
            # Fill weighted histogram
            temp_weight = r_weight[0][i]*temp_points[0][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            data_rand[0] += temp_hist
            # Fill unweighted histogram
            temp_weight = r_weight[1][i]*temp_points[1][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            data_rand[1] += temp_hist

        return data_rand, self.__bins_s

    def pairs_separation(self, no_job, total_jobs, out="DD", leaf=40):
        """ Calculate separation distribution between pairs of galaxies.
        Use a modfied nearest-neighbors k-d tree algorithm to calculate distance
        up to a given radius defined in config file.
        Metric: Euclidean (or Minkowski with p=2).
        Inputs:
        + no_job: int
            Job number. Must be greater or equal to 0 and less than total_jobs
            (0 <= no_job < total_jobs).
        + total_jobs: int
            Total number of jobs.
        + out: string (default="DD")
            Valid argument are "RR", "DR", DD". Distribution to calculate.
        + leaf: int (default=40)
            Number of points at which to switch to brute-force. For a specified
            leaf_size, a leaf node is guaranteed to satisfy
            leaf_size <= n_points <= 2*leaf_size, except in the case that
            n_samples < leaf_size. More details in sklearn.neighbors.KDTree.
        Outputs:
        + pairs_separation: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted pairs_separation
            respectively.
        + bins: array
            Binedges of DD(s) (length(data_data_hist)+1).
        """
        if self.data_cat is None or self.rand_cat is None:
            raise TypeError("Catalogs are not imported.")

        # Choose catalogs based on input
        if out == "DD":
            tree_cat = self.data_cat
            point_cat = self.data_cat
        elif out == "RR":
            tree_cat = self.rand_cat
            point_cat = self.rand_cat
        elif out == "DR":
            # Optimizing: Run time of k-d tree modified nearest-neighbors is
            # O(NlogM) where M is the number of points in Tree and N is the
            # number of points for pairings. Thus, the kd-tree is created using
            # the smaller of galaxies catalog and randoms catalog.
            if self.data_cat.shape[0] > self.rand_cat.shape[0]:
                tree_cat = self.data_cat
                point_cat = self.rand_cat
            else:
                tree_cat = self.rand_cat
                point_cat = self.data_cat

        # Convert Celestial coordinate into Cartesian coordinate and create
        # k-d tree and point cartesian catalog.
        # Point cartesian catalog
        temp = celestial2cart(point_cat[:, 0], point_cat[:, 1], point_cat[:, 2])
        cart_point_cat = numpy.array([temp[0], temp[1], temp[2],
                                      point_cat[:, 3]]).T

        # k-d tree
        temp = celestial2cart(tree_cat[:, 0], tree_cat[:, 1], tree_cat[:, 2])
        cart_tree_cat = numpy.array([temp[0], temp[1], temp[2],
                                     tree_cat[:, 3]]).T
        tree = KDTree(cart_tree_cat[:, :3], leaf_size=leaf, metric='euclidean')

        # Calculate start and end index based on job number and total number of
        # jobs.
        job_range = get_job_index(no_job, total_jobs, cart_point_cat.shape[0])

        # Compute pairs separation using modified nearest-neighbors algorithm
        # to calculate angular distance up to a given radius.
        pairs_separation = self.__pairs_separation_thread(cart_point_cat,
                                                          cart_tree_cat, tree,
                                                          *job_range)

        return pairs_separation, self.__bins_s

    def correlation(self, rand_rand, data_rand, data_data, bins):
        """ Construct two-point correlation function.
        Inputs:
        + rand_rand: array
            Values of separation distribution between pairs of random galaxies
            RR(s).
        + data_rand: array
            Values of separation distribution between pairs of a random point
            and a galaxy DR(s).
        + data_data: array
            Valus of separation distribution between pairs of galaxies DD(s).
        Note: rand_rand, data_rand, data_data must have the same size.
        + bins: array
            Binedges of rand_rand, data_rand, data_data.
        Output:
        + correlation_function: array
            Two-point correlation function computed using equations:
            f = [DD(s)-2*DR(s)+RR(s)]/RR(s)
            If RR(s) = 0, then f = 0
        + correlation_function_ss: array
            Two-point correlation function multiplied by s^2: f(s)s^2
            """
        correlation_function = data_data-2*data_rand+rand_rand
        correlation_function = numpy.divide(correlation_function, rand_rand,
                                            out=numpy.zeros_like(rand_rand),
                                            where=rand_rand != 0)
        center_s = 0.5*(bins[:-1]+bins[1:])
        correlation_function_ss = correlation_function*center_s**2
        return correlation_function, correlation_function_ss

    # Compute normalization constant
    def normalization(self, weighted=True):
        """ Calculate and return normalization factor
        Inputs:
        + weighted: bool (default=True)
            If True, return weighted normalization constants.
            If False, return unweighted normalization constants.
        Outputs:
        + norm_rr: int, float
            Normalization constant for RR(s)
        + norm_dr: int, float
            Normalization constant for DR(s)
        + norm_rr: int, float
            Normalization constant for DD(s)
        Note:
        Equations for unweighted normalization constant:
        - norm_rr = 0.5*n_rand*(n_rand-1)
        - norm_dd = 0.5*n_data*(n_data-1)
        - norm_dr = n_rand*n_data
        where n_rand and n_data are the size of the randoms and galaxies
        catalog respectively.
        Equations for weighted normalization constant:
        - norm_rr = 0.5*(sum_w_rand^2-sum_w2_rand)
        - norm_dd = 0.5*(sum_w_data^2-sum_w2_rand)
        - norm_dr = sum_w_data*sum_w_rand
        where sum_w_rand and sum_w_data is the sum of weights, sum_w2_rand
        and sum_w2_data is the sum of weights squared in randoms and galaxies
        catalog respectively.
        """
        if self.data_cat is None or self.rand_cat is None:
            raise TypeError("Catalogs are not imported.")

        if weighted:
            # Calculate weighted normalization constant
            w_sum_rand = numpy.sum(self.rand_cat[:, 3])
            w_sum_data = numpy.sum(self.data_cat[:, 3])
            w2_sum_rand = numpy.sum(self.rand_cat[:, 3]**2)
            w2_sum_data = numpy.sum(self.data_cat[:, 3]**2)

            norm_rr = 0.5*(w_sum_rand**2-w2_sum_rand)
            norm_dd = 0.5*(w_sum_data**2-w2_sum_data)
            norm_dr = w_sum_rand*w_sum_data

            return norm_rr, norm_dr, norm_dd

        # Calculate unweighted normalization constant
        n_rand = self.rand_cat.shape[0]
        n_data = self.data_cat.shape[0]

        norm_rr = 0.5*n_rand*(n_rand-1)
        norm_dd = 0.5*n_data*(n_data-1)
        norm_dr = n_data*n_rand

        return norm_rr, norm_dr, norm_dd

    def get_error(self, hist_w, hist_u):
        """ Get the bin error of weighted and unweighted pairs separation
        histogram. Bin error in unweighted pairs separation is assumed to be
        Poisson. Bin error in weighted pairs separation is given by equation
        error_w = n_w/sqrt(n_u).
        Inputs:
        + hist_w: array
            Values of weighted histogram.
        + hist_u: array
            Values of unweighted histogram.
        Outputs:
        + error: ndarray or tuples of ndarray
            The bun error in the weighted and unweighted histograms respectively.
        """
        error_u = numpy.sqrt(hist_u)
        error_w = numpy.divide(hist_w, numpy.sqrt(hist_u),
                               out=numpy.zeros_like(hist_w),
                               where=hist_u != 0)
        error = numpy.array([error_w, error_u]).T
        return error
