""" Unittest for correlation_function.py """


import unittest2
import numpy
import correlation_function

PI = 3.14159265359


class TestCorrelationFunction(unittest2.TestCase):
    """ Test class for module correlation_function """
    def test_constant(self):
        """ Test function for module constant """
        print("\nTest global constants")
        print("- DEG2RAD")
        self.assertAlmostEqual(correlation_function.DEG2RAD, 0.0174532925199)
        print("- RAD2DEG")
        self.assertAlmostEqual(correlation_function.RAD2DEG, 57.2957795131)

    def test_get_distance(self):
        """ Test function correlation_function.get_distance(r1, r2, theta) """
        print("\nTest get_distance")
        funct = correlation_function.get_distance

        # Commutative
        print("- Commutative test ")
        self.assertAlmostEqual(funct(3., 5., 1.3), 5.09657091987)
        self.assertAlmostEqual(funct(5., 3., 1.3), 5.09657091987)

        # Rotate by 360 deg (or 2pi rad)
        print("- Rotation 360 degree test")
        self.assertAlmostEqual(funct(3., 5., 1.3+2*PI),
                               5.09657091987)

        # Negative radius (Rotate by 180 deg)
        print("- Rotation 180 degreee test (negative radius)")
        self.assertAlmostEqual(funct(3., 5., 1.3+PI), 6.48266649294)
        self.assertAlmostEqual(funct(-3., 5., 1.3), 6.48266649294)
        self.assertAlmostEqual(funct(3., -5., 1.3), 6.48266649294)

        # Distance to origin
        print("- Distance to origin test")
        self.assertEqual(funct(3., 0., 0.8), 3.)
        self.assertEqual(funct(0., 5., 0.8), 5.)

        # Same points
        print("- Same points test")
        self.assertAlmostEqual(funct(3., 3., 0.), 0.)

    def test_get_bins(self):
        """ Test function correlation_function.get_binnings(min, max, width) """
        print("\nTest get_binnings")
        funct = correlation_function.get_bins

        # Range is a factor of binwidth
        print("- Range is a factor of binwidth test")
        true_array = numpy.array([6.4, 6.9, 7.4, 7.9])
        test_array, test_binwidth = funct(6.4, 7.9, 0.5)
        numpy.testing.assert_almost_equal(test_array, true_array)
        self.assertAlmostEqual(test_binwidth, 0.5)

        # Range is not a factor of binwidth
        print("- Range is a not factor of binwidth test")
        true_array = numpy.array([6.4, 6.78, 7.16, 7.54, 7.92])
        test_array, test_binwidth = funct(6.4, 7.92, 0.5)
        numpy.testing.assert_almost_equal(test_array, true_array)
        self.assertAlmostEqual(test_binwidth, 0.38)

        # Range goes from negative to postive
        print("- Range goes from negative to positive test")
        true_array = numpy.array([-1.3, -0.77, -0.24, 0.29, 0.82])
        test_array, test_binwidth = funct(-1.3, 0.82, 0.65)
        numpy.testing.assert_almost_equal(test_array, true_array)
        self.assertAlmostEqual(test_binwidth, 0.53)

        # Min is greater than Max. Check for exception
        print("- Min is greater than Max test")
        with self.assertRaises(Exception) as context:
            funct(1.3, 0.5, 0.2)
        self.assertTrue("Max must be greater than Min." in
                        str(context.exception))

    def test_hist2point(self):
        """ Test function correlation_function.hist2point() """
        print("\nTest hist2point")
        funct = correlation_function.hist2point

        # General case
        print("- General case test")
        x_edges = numpy.array([-4.7, -4.3, -3.9, -3.5])
        y_edges = numpy.array([2.5, 3.5, 4.5, 5.5])
        hist = numpy.array([[0, -2.4, 1.2], [6.5, 2., 3.3], [0., 0., 5.2]])
        true_array = numpy.array([[-4.1, 3.0, 6.5],
                                  [-4.5, 4.0, -2.4],
                                  [-4.1, 4.0, 2.0],
                                  [-4.5, 5.0, 1.2],
                                  [-4.1, 5.0, 3.3],
                                  [-3.7, 5.0, 5.2]])
        numpy.testing.assert_almost_equal(funct(hist, x_edges, y_edges),
                                          true_array)

        # General case without excluding zeros
        print("- General case without excluding zeros test")
        true_array = numpy.array([[-4.5, 3.0, 0.],
                                  [-4.1, 3.0, 6.5],
                                  [-3.7, 3.0, 0.],
                                  [-4.5, 4.0, -2.4],
                                  [-4.1, 4.0, 2.0],
                                  [-3.7, 4.0, 0.],
                                  [-4.5, 5.0, 1.2],
                                  [-4.1, 5.0, 3.3],
                                  [-3.7, 5.0, 5.2]])
        numpy.testing.assert_almost_equal(funct(hist, x_edges, y_edges, False),
                                          true_array)

        # Array with all zeros
        print("- All zeros test")
        hist = numpy.zeros_like(hist)
        numpy.testing.assert_almost_equal(funct(hist, x_edges, y_edges),
                                          numpy.array([]).reshape(0, 3))

        # Array with most zeros
        print("- Most zeros test")
        hist = numpy.zeros_like(hist)
        hist[1][2] = 3.1
        hist[0][1] = 2.3
        true_array = numpy.array([[-4.5, 4.0, 2.3],
                                  [-4.1, 5.0, 3.1]])
        numpy.testing.assert_almost_equal(funct(hist, x_edges, y_edges),
                                          true_array)

    def test_correlation_class(self):
        """ Test correlation_function.CorrelationFunction class """
        print("\nTest CorrelationFunction")

        test = numpy.load("test/tpcf_test.npz")
        tpcf = correlation_function.CorrelationFunction("test/test_config.cfg")

        # Test normalization
        print(" - Normalization test")
        norm = numpy.array([tpcf.normalization(weighted=True),
                            tpcf.normalization(weighted=False)])
        numpy.testing.assert_almost_equal(norm, test["NORM"])

        # Test comoving distribution P(r)
        print(" - Comoving distribution test ")
        r_hist, bins_r = tpcf.comoving_distribution()
        numpy.testing.assert_almost_equal(r_hist, test["R_HIST"])
        numpy.testing.assert_almost_equal(bins_r, test["BINS_PR"])

        # Test angular distribution f(theta)
        print(" - Angular distribution f(theta) test ")
        theta_hist, bins_theta = tpcf.angular_distance(0, 1)
        numpy.testing.assert_almost_equal(theta_hist, test["ANGULAR_D"])
        numpy.testing.assert_almost_equal(bins_theta, test["BINS_FTHETA"])

        # Test angular-radial distribution g(theta, r)
        print(" - Angular-radial distribution g(theta, r) test ")
        theta_r_hist, bins_theta, bins_r = tpcf.angular_comoving(0, 1)
        numpy.testing.assert_almost_equal(theta_r_hist, test["ANGULAR_R"])
        numpy.testing.assert_almost_equal(bins_theta, test["BINS_GTHETA"])
        numpy.testing.assert_almost_equal(bins_r, test["BINS_GR"])

        # Test random-random distribution RR(s)
        print(" - Random-random distribution RR(s) test")
        rand_rand, bins_s = tpcf.rand_rand(theta_hist, r_hist)
        numpy.testing.assert_almost_equal(rand_rand, test["RR"])
        numpy.testing.assert_almost_equal(bins_s, test["BINS_RR"])

        # Test data-random distribution DR(s)
        print(" - Data-random distribution DR(s) test")
        data_rand, bins_s = tpcf.data_rand(theta_r_hist, r_hist)
        numpy.testing.assert_almost_equal(data_rand, test["DR"])
        numpy.testing.assert_almost_equal(bins_s, test["BINS_DR"])

        # Test data-data distribution DD(s)
        print(" - Data-data distribution DD(s) test")
        data_data, bins_s = tpcf.pairs_separation(0, 1, out="DD")
        numpy.testing.assert_almost_equal(data_data, test["DD"])
        numpy.testing.assert_almost_equal(bins_s, test["BINS_DD"])

        # Test correlation function
        print(" - Correlation test")
        correlation = tpcf.correlation(rand_rand[0], data_rand[0], data_data[0],
                                       bins_s)
        numpy.testing.assert_almost_equal(correlation, test["TPCF"])


def main():
    """ Run unittest """
    unittest2.main()


if __name__ == "__main__":
    main()
