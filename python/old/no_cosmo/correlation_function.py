""" Modules to construct two-point correlation function using KD-Tree
Estimators used: (DD-2*DR+RR)/RR,
where D and R are data and randoms catalogs respectively """
__version__ = 1.11

import configparser
import numpy
from sklearn.neighbors import KDTree, BallTree
from astropy.io import fits
from cosmology import Cosmology

DEG2RAD = numpy.pi/180.
RAD2DEG = 180./numpy.pi


def import_fits(fname_key, fits_reader, cosmo):
    """ Import data into 2-d array
    Inputs:
    + fname_key: string
        Key for data filename in reader.
    + fits_readers: dict
        Must have attributes: "INDEX"=index of headers. RA", "DEC", "Z",
        "WEIGHT"=corresponded variable names in header.
    + cosmol: cosmology.Cosmology
        Cosmological parameters to convert redshift to comoving distance.
    Outputs:
    + catalog: ndarray or tuple of ndarrays
        Return catalog format in each row [DEC, RA, R, WEIGHT].
    """
    header_index = int(fits_reader["INDEX"])
    hdulist = fits.open(fits_reader[fname_key])
    tbdata = hdulist[header_index].data
    temp_dec = DEG2RAD*tbdata[fits_reader["DEC"]]
    temp_ra = DEG2RAD*tbdata[fits_reader["RA"]]
    temp_r = cosmo.z2r(tbdata[fits_reader["Z"]])
    try:
        temp_weight_fkp = tbdata[fits_reader["WEIGHT_FKP"]]
        temp_weight_noz = tbdata[fits_reader["WEIGHT_NOZ"]]
        temp_weight_cp  = tbdata[fits_reader["WEIGHT_CP"]]
        temp_weight_sdc = tbdata[fits_reader["WEIGHT_SDC"]]
        temp_weight = (temp_weight_sdc*temp_weight_fkp*
                       (temp_weight_noz + temp_weight_cp -1))
    except KeyError:
        temp_weight = tbdata[fits_reader["WEIGHT"]]
    catalog = numpy.array([temp_dec, temp_ra, temp_r, temp_weight]).T
    hdulist.close()
    return catalog


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


def get_binnings(x_min, x_max, binwidth):
    """ Return the binnings given min, max and width """
    nbins = int(numpy.ceil((x_max-x_min)/binwidth))
    return numpy.linspace(x_min, x_max, nbins+1)


class CorrelationFunction():
    """ Class to construct two-point correlation function """
    def __init__(self, config_fname):
        """ Constructor takes in configuration file and sets up binning
        variables """
        config = configparser.ConfigParser()
        config.read(config_fname)

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
        reader = config['FITS']
        self.data_cat = import_fits('data_filename', reader, cosmo)
        self.rand_cat = import_fits('random_filename', reader, cosmo)

        # Setting up binning variables
        binnings = config['BINNING']
        binwidth_ra = DEG2RAD*float(binnings['binwidth_ra'])
        binwidth_dec = DEG2RAD*float(binnings['binwidth_dec'])
        # binwidth of angular distance distribution
        binwidth_theta = DEG2RAD*float(binnings['binwidth_theta'])
        binwidth_r = float(binnings['binwidth_r'])
        binwidth_s = float(binnings['binwidth_s'])

        ra_min = min(self.rand_cat[:, 1].min(), self.data_cat[:, 1].min())
        ra_max = max(self.rand_cat[:, 1].max(), self.data_cat[:, 1].max())
        dec_min = min(self.rand_cat[:, 0].min(), self.data_cat[:, 0].min())
        dec_max = max(self.rand_cat[:, 0].max(), self.data_cat[:, 0].max())
        r_min = min(self.rand_cat[:, 2].min(), self.data_cat[:, 2].min())
        r_max = max(self.rand_cat[:, 2].max(), self.data_cat[:, 2].max())
        s_max = float(binnings['s_max'])
        # maximum angular distance to be considered between any given points
        theta_max = numpy.arccos(1.-0.5*s_max**2/r_min**2)

        self.__bins_ra = get_binnings(ra_min, ra_max, binwidth_ra)
        self.__bins_dec = get_binnings(dec_min, dec_max, binwidth_dec)
        self.__bins_r = get_binnings(r_min, r_max, binwidth_r)
        self.__bins_s = get_binnings(0., s_max, binwidth_s)
        self.__bins_theta = get_binnings(0., theta_max, binwidth_theta)

    def __get_error(self, hist):
        """ Return the uncertainty in unnormalized DD(s), DR(s), and RR(s). The
        unweighted uncertainty is assumed to be Poisson. The weighted uncertainty
        is calculated as sigma_w = hist_w/sigma_u where hist_w is the value of
        the histogram. If sigma_uw is zero, then sigma_w is also zero.
        Inputs:
        + hist: ndarray or tuples of ndarray
            Values of weighted and unweighted DD(s), DR(s), or RR(s).
            weight=hist[0], unweight=hist[1]
        Outputs:
        + error_hist: ndarray or tuples of ndarray
            Errors of weighted and unweighted hist. weight=error_hist[0],
            unweight=error_hist[1]
            """
        error_hist_u = numpy.sqrt(hist[1])
        error_hist_w = numpy.divide(hist[0], error_hist_u,
                                    out=numpy.zeros_like(hist[0]),
                                    where=error_hist_u!=0)
        error_hist = numpy.array([error_hist_w, error_hist_u])
        return error_hist

    def angular_distance(self, leaf=40):
        """ Construct f(theta), the angular distance distribution, as an
        one-dimensional histogram. Binnings are defined in config file.
        Use a modified nearest-neighbors BallTree algorithm to calculate
        angular distance up to a given radius defined in config file.
        Inputs:
        + leaf: int
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
        # Compute 2d angular distribution R(ra, dec) and breaks them into data
        # points with proper weights.
        angular_hist = numpy.histogram2d(self.rand_cat[:, 0],
                                         self.rand_cat[:, 1],
                                         bins=(self.__bins_dec, self.__bins_ra))
        angular = hist2point(*angular_hist)

        # Create a BallTree and use modified nearest-neighbors algorithm to
        # calculate angular distance up to a given radius.
        arc_tree = BallTree(angular[:, :2], leaf_size=leaf, metric='haversine')

        # Define angular distance distribution f(theta) as histogram
        nbins_theta = self.__bins_theta.size-1
        theta_max = self.__bins_theta.max()
        theta_hist = numpy.zeros(nbins_theta)

        print("Construct f(theta)")
        for i, point in enumerate(angular):
            if i % 10000 is 0:
                print(i)
            index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                                 r=theta_max,
                                                 return_distance=True)
            temp_weight = point[2]*angular[:, 2][index[0]]
            temp_hist, _ = numpy.histogram(theta[0], bins=nbins_theta,
                                           range=(0., theta_max),
                                           weights=temp_weight)
            theta_hist += temp_hist
        # Correction for double counting
        theta_hist = theta_hist/2.

        return theta_hist, self.__bins_theta

    def r_angular_distance(self, leaf=40):
        """ Construct g(theta, z), angular distance vs. redshift distribution,
        as a two-dimensional histogram. Binnings are defined in config file.
        Use a modified nearest-neighbors BallTree algorithm to calculate
        angular distance up to a given radius defined in config file.
        Inputs:
        + leaf: int
            Number of points at which to switch to brute-force. For a specified
            leaf_size, a leaf node is guaranteed to satisfy
            leaf_size <= n_points <= 2*leaf_size, except in the case that
            n_samples < leaf_size. More details in sklearn.neighbors.BallTree.
        Outputs:
        + r_theta_hist: ndarray or tuple of ndarrays
            Return values of weighted and unweighted g(theta, r) respectively.
            Each has dimension (length(bins_theta)-1, length(bins_r)-1).
        + bins_theta: array
            Binedges along x-axis.
        + bins_r: array
            Binedges along y-axis.
        """
        # Compute 2d angular distribution R(ra, dec) and breaks them into data
        # points with proper weights.
        angular_hist = numpy.histogram2d(self.rand_cat[:, 0],
                                         self.rand_cat[:, 1],
                                         bins=(self.__bins_dec, self.__bins_ra))
        angular = hist2point(*angular_hist)

        # Define angular-radial distribution g(theta, r) as weighted and
        # unweighted 2-d histogram respectively
        nbins_theta = self.__bins_theta.size-1
        nbins_r = self.__bins_r.size-1
        theta_max = self.__bins_theta.max()
        bins_range = ((0., theta_max),
                      (self.__bins_r.min(), self.__bins_r.max()))
        r_theta_hist = numpy.zeros((2, nbins_theta, nbins_r))

        # Optimizing: Runtime of BallTree modified nearest-neighbors is O(NlogM)
        # where M is the number of points in Tree and N is the number of points
        # for pairings. Thus, the BallTree is created using the size of the
        # smaller catalog, galaxies catalog vs. angular catalog.
        if angular.shape[0] >= self.data_cat.shape[0]:
            arc_tree = BallTree(angular[:, :2], leaf_size=leaf,
                                metric='haversine')

            print("Construct g(theta, r)")
            for i, point in enumerate(self.data_cat):
                if i % 10000 is 0:
                    print(i)
                index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                                     r=theta_max,
                                                     return_distance=True)
                temp_r = numpy.repeat(point[2], index[0].size)
                # Fill unweighted histogram
                temp_weight = angular[:, 2][index[0]]
                temp_hist, _, _ = numpy.histogram2d(theta[0], temp_r,
                                                    bins=(nbins_theta, nbins_r),
                                                    range=bins_range,
                                                    weights=temp_weight)
                r_theta_hist[1] += temp_hist
                # Fill weighted histogram
                temp_weight = temp_weight*point[3]
                temp_hist, _, _ = numpy.histogram2d(theta[0], temp_r,
                                                    bins=(nbins_theta, nbins_r),
                                                    range=bins_range,
                                                    weights=temp_weight)
                r_theta_hist[0] += temp_hist
        else:
            arc_tree = BallTree(self.data_cat[:, :2], leaf_size=leaf,
                                metric='haversine')

            print("Construct g(theta, r)")
            for i, point in enumerate(angular):
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
                r_theta_hist[0] += temp_hist
                # Fill unweighted histogram
                temp_weight = numpy.repeat(point[2], index[0].size)
                temp_hist, _, _ = numpy.histogram2d(theta[0], temp_r,
                                                    bins=(nbins_theta, nbins_r),
                                                    range=bins_range,
                                                    weights=temp_weight)
                r_theta_hist[1] += temp_hist

        return r_theta_hist, self.__bins_theta, self.__bins_r


    def rand_rand(self, leaf=40):
        """ Construct separation distribution RR(s) between pairs of randoms.
        Inputs:
        + leaf: int
            Number of points at which to switch to brute-force. For a specified
            leaf_size, a leaf node is guaranteed to satisfy
            leaf_size <= n_points <= 2*leaf_size, except in the case that
            n_samples < leaf_size. More details in sklearn.neighbors.BallTree.
        Outputs:
        + rand_rand: ndarrays or tupple of ndarrays
            Return values of weighted and unweighted RR(s) respectively.
        + bins: array
            Binedges of RR(s) (length(rand_rand_hist)+1).
        """
        # Construct weighted and unweighted radial distribution P(r) as two
        # one-dimensional histograms respectively.
        r_hist = numpy.zeros((2, self.__bins_r.size-1))
        r_hist[0] += numpy.histogram(self.rand_cat[:, 2], bins=self.__bins_r,
                                     weights=self.rand_cat[:, 3])[0]
        r_hist[1] += numpy.histogram(self.rand_cat[:, 2], bins=self.__bins_r)[0]
        r_hist = 1.*r_hist/self.rand_cat.shape[0]

        # Construct angular separation distribuion f(theta)
        theta_hist, _ = self.angular_distance(leaf)

        # Convert f(theta) and P(r) into data points, weighted and unweighted
        weight = numpy.zeros((2, theta_hist.size, r_hist.shape[1]))
        for i in range(theta_hist.size):
            for j in range(r_hist.shape[1]):
                weight[0][i, j] = theta_hist[i]*r_hist[0][j]
                weight[1][i, j] = theta_hist[i]*r_hist[1][j]
        temp_points = [hist2point(weight[0], self.__bins_theta, self.__bins_r),
                       hist2point(weight[1], self.__bins_theta, self.__bins_r)]
        center_r = 0.5*(self.__bins_r[:-1]+self.__bins_r[1:])
        center_r = center_r[r_hist[1] != 0]

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
            temp_weight = r_hist[0][i]*temp_points[0][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, bins=nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            rand_rand[0] += temp_hist
            # Fill unweighted histogram
            temp_weight = r_hist[1][i]*temp_points[1][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, bins=nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            rand_rand[1] += temp_hist

        # get error
        error_rand_rand = self.__get_error(rand_rand)

        return rand_rand, error_rand_rand, self.__bins_s

    def data_rand(self, leaf=40):
        """ Construct separation distribution DR(s) between pairs of a random
        point and a galaxy.
        Inputs:
        + leaf: int
            Number of points at which to switch to brute-force. For a specified
            leaf_size, a leaf node is guaranteed to satisfy
            leaf_size <= n_points <= 2*leaf_size, except in the case that
            n_samples < leaf_size. More details in sklearn.neighbors.BallTree.
        Outputs:
        + data_rand: ndarrays or tupple of ndarrays
            Return values of weighted and unweighted DR(s) respectively.
        + bins: array
            Binedges of DR(s) (length(data_rand_hist)+1).
        """
        # Construct weighted and unweighted radial distribution P(r) as two
        # one-dimensional histograms respectively.
        r_hist = numpy.zeros((2, self.__bins_r.size-1))
        r_hist[0] += numpy.histogram(self.rand_cat[:, 2], bins=self.__bins_r,
                                     weights=self.rand_cat[:, 3])[0]
        r_hist[1] += numpy.histogram(self.rand_cat[:, 2], bins=self.__bins_r)[0]
        r_hist = 1.*r_hist/self.rand_cat.shape[0]

        # Construct weighted and unweighted radial angular separation
        # distribution g(theta, r) as two 2-d histograms respectively.
        r_theta_hist, _, _ = self.r_angular_distance(leaf)
        # Convert g(theta, r) into data points, weighted and unweighted
        temp_points = (
            (hist2point(r_theta_hist[0], self.__bins_theta, self.__bins_r),
             hist2point(r_theta_hist[1], self.__bins_theta, self.__bins_r)))

        # Define weighted and unweighted DR(s) as two one-dimensional
        # histograms respectively.
        nbins_s = self.__bins_s.size-1
        s_max = self.__bins_s.max()
        data_rand = numpy.zeros((2, nbins_s))

        # Integration
        print("Construct DR(s)")
        center_r = 0.5*(self.__bins_r[:-1]+self.__bins_r[1:])
        for i, temp_r in enumerate(center_r[r_hist[1] != 0]):
            if i % 100 is 0:
                print(i)
            temp_s = get_distance(temp_r, temp_points[0][:, 1],
                                  temp_points[0][:, 0])
            # Fill weighted histogram
            temp_weight = r_hist[0][i]*temp_points[0][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            data_rand[0] += temp_hist
            # Fill unweighted histogram
            temp_weight = r_hist[1][i]*temp_points[1][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            data_rand[1] += temp_hist

        # get error
        error_data_rand = self.__get_error(data_rand)

        return data_rand,  error_data_rand, self.__bins_s

    def data_data(self, leaf=40):
        """ Construct separation distribution DD(s) between pairs of galaxies.
        Use a modified nearest-neighbors KDTree algorithm to calculate distance
        up to a given radius defined in config file. Metric: Euclidean (or
        Minkowski with p=2).
        Inputs:
        + leaf: int
            Number of points at which to switch to brute-force. For a specified
            leaf_size, a leaf node is guaranteed to satisfy
            leaf_size <= n_points <= 2*leaf_size, except in the case that
            n_samples < leaf_size. More details in sklearn.neighbors.KDTree.
        Outputs:
        + data_data: ndarrays or tupple of ndarrays
            Return values of weighted and unweighted DD(s) respectively.
        + bins: array
            Binedges of DD(s) (length(data_data_hist)+1).
        """
        # First convert into Cartesian coordinates
        # Conversion equation is
        # x = r*cos(dec)*cos(ra)
        # y = r*cos(dec)*sin(ra)
        # z = r*sin(dec)
        temp_ra = self.data_cat[:, 1]
        temp_dec = self.data_cat[:, 0]
        temp_r = self.data_cat[:, 2]
        temp_weight = self.data_cat[:, 3]
        temp_x = temp_r*numpy.cos(temp_dec)*numpy.cos(temp_ra)
        temp_y = temp_r*numpy.cos(temp_dec)*numpy.sin(temp_ra)
        temp_z = temp_r*numpy.sin(temp_dec)
        cart_cat = numpy.array([temp_x, temp_y, temp_z, temp_weight]).T

        # Create KD-tree and compute DD(s) using modified nearest-neighbors
        # algorithm to calculate angular distance up to a given radius.
        cart_tree = KDTree(cart_cat[:, :3], leaf_size=leaf, metric='euclidean')

        # Define weighted and unweighted DD(s) as two one-dimensional
        # histograms respectively.
        nbins_s = self.__bins_s.size-1
        s_max = self.__bins_s.max()
        data_data = numpy.zeros((2, nbins_s))

        print("Start constructing DD(s)...")
        for i, point in enumerate(cart_cat):
            if i % 10000 is 0:
                print(i)
            index, dist = cart_tree.query_radius(point[: 3].reshape(1, -1),
                                                 r=s_max,
                                                 return_distance=True)
            # Fill weighted histogram
            temp_weight = cart_cat[:, 3][index[0]]*cart_cat[i, 3]
            temp_hist, _ = numpy.histogram(dist[0], bins=nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            data_data[0] += temp_hist
            # Fill unweighted histogram
            temp_hist, _ = numpy.histogram(dist[0], bins=nbins_s,
                                           range=(0., s_max))
            data_data[1] += temp_hist

        # correction for double counting
        data_data[0][0] -= numpy.sum(cart_cat[:, 3]**2)  # sum of weight square
        data_data[1][0] -= self.data_cat.shape[0]
        data_data = data_data/2.

        # get error
        error_data_data = self.__get_error(data_data)

        return data_data, error_data_data, self.__bins_s

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
            Valus of separation distribution between paris of galaxies DD(s).
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
        """ Compute and return normalization factor
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
        if weighted:
            # Compute weighted normalization constant
            w_sum_rand = numpy.sum(self.rand_cat[:, 3])
            w_sum_data = numpy.sum(self.data_cat[:, 3])
            w2_sum_rand = numpy.sum(self.rand_cat[:, 3]**2)
            w2_sum_data = numpy.sum(self.data_cat[:, 3]**2)

            norm_rr = 0.5*(w_sum_rand**2-w2_sum_rand)
            norm_dd = 0.5*(w_sum_data**2-w2_sum_data)
            norm_dr = w_sum_rand*w_sum_data

            return norm_rr, norm_dr, norm_dd

        # Compute unweighted normalization constant
        n_rand = self.rand_cat.shape[0]
        n_data = self.data_cat.shape[0]

        norm_rr = 0.5*n_rand*(n_rand-1)
        norm_dd = 0.5*n_data*(n_data-1)
        norm_dr = n_data*n_rand

        return norm_rr, norm_dr, norm_dd
