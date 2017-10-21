""" Unit test for correlation_function.py """


import unittest2
import numpy
import correlation_function
import cosmology

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


    def test_get_binnings(self):
        """ Test function correlation_function.get_binnings(min, max, width) """
        print("\nTest get_binnings")
        funct = correlation_function.get_binnings

        # Range is a factor of binwidth
        print("- Range is a factor of binwidth test")
        true_array = numpy.array([6.4, 6.9, 7.4, 7.9])
        numpy.testing.assert_almost_equal(funct(6.4, 7.9, 0.5), true_array)

        # Range is not a factor of binwidth
        print("- Range is a not factor of binwidth test")
        true_array = numpy.array([6.4, 6.78, 7.16, 7.54, 7.92])
        numpy.testing.assert_almost_equal(funct(6.4, 7.92, 0.5), true_array)

        # Range goes from negative to postive
        print("- Range goes from negative to positive test")
        true_array = numpy.array([-1.3, -0.77, -0.24, 0.29, 0.82])
        numpy.testing.assert_almost_equal(funct(-1.3, 0.82, 0.65), true_array)

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
        print("- General case")
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
        print("- General calse without excluding zeros")
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
        print("- All zeros")
        hist = numpy.zeros_like(hist)
        numpy.testing.assert_almost_equal(funct(hist, x_edges, y_edges),
                                          numpy.array([]).reshape(0, 3))

        # Array with most zeros
        print("- Most zeros")
        hist = numpy.zeros_like(hist)
        hist[1][2] = 3.1
        hist[0][1] = 2.3
        true_array = numpy.array([[-4.5, 4.0, 2.3],
                                  [-4.1, 5.0, 3.1]])
        numpy.testing.assert_almost_equal(funct(hist, x_edges, y_edges),
                                          true_array)

    def test_correlation_class(self):
        """ Test class correlation function """
        # import unittest data
        test = numpy.load("unittest/tpcf_test.npz")

        cosmo = cosmology.Cosmology()

        tpcf = correlation_function.CorrelationFunction("unittest/config_test.cfg")
        norm = numpy.array([tpcf.normalization(weighted=True),
                            tpcf.normalization(weighted=False)])
        rand_rand, err_rand_rand, bins_rand_rand = tpcf.rand_rand(cosmo)
        data_rand, err_data_rand, bins_data_rand = tpcf.data_rand(cosmo)
        data_data, err_data_data, bins_data_data = tpcf.data_data(cosmo)
        correlation = tpcf.correlation(rand_rand[0], data_rand[0], data_data[0],
                                       bins_rand_rand)

        print(" - Normalization")
        numpy.testing.assert_almost_equal(norm, test["NORM"])

        print(" - Random-Random RR(s)")
        numpy.testing.assert_almost_equal(rand_rand, test["RR"])
        numpy.testing.assert_almost_equal(err_rand_rand, test["E_RR"])
        numpy.testing.assert_almost_equal(bins_rand_rand, test["BINS"])

        print(" - Data-Random DR(s)")
        numpy.testing.assert_almost_equal(data_rand, test["DR"])
        numpy.testing.assert_almost_equal(err_data_rand, test["E_DR"])
        numpy.testing.assert_almost_equal(bins_data_rand, test["BINS"])

        print(" - Data-Data DD(s)")
        numpy.testing.assert_almost_equal(data_data, test["DD"])
        numpy.testing.assert_almost_equal(err_data_data, test["E_DD"])
        numpy.testing.assert_almost_equal(bins_data_data, test["BINS"])

        print(" - Correlation")
        numpy.testing.assert_almost_equal(correlation, test["TPCF_W"])


def main():
    unittest2.main()


if __name__ == "__main__":
    main()