import unittest
from hist_fast import Hist1D, Hist2D
import numpy as np
from subprocess import call

class Hist1DTestCase(unittest.TestCase):
    """Tests for `Hist1D`."""

    def test_init(self):
        h = Hist1D(5, 0., 1.)
        self.assertEqual(h.nxbins, 5)
        self.assertEqual(h.xmin, 0.)
        self.assertEqual(h.xmax, 1.)
        self.assertTrue(np.array_equal(h.xedges, np.linspace(0., 1., 6)))

    def test_fill(self):
        # Fill bin 0
        h = Hist1D(5, 0., 1.)
        h.fill(0.1, 1.5)
        self.assertEqual(h.getBinEntries(0), 1)
        self.assertEqual(h.getBinValue(0), 1.5)
        self.assertEqual(h.getBinErr(0), 0.)
        self.assertEqual(h.getBinErr(3), 0.)

        # Fill bin 0 again
        h.fill(0.1, 3)
        self.assertEqual(h.getBinEntries(0), 2)
        self.assertEqual(h.getBinValue(0), 4.5)
        self.assertTrue(np.isclose(h.getBinErr(0), 1.06066017178))

        # Overflow/underflow: do nothing
        h.fill(10.)
        self.assertEqual(h.getBinEntries(4), 0)
        h.fill(-10.)
        self.assertEqual(h.getBinEntries(0), 2)

    def test_bin_mean(self):
        h = Hist1D(5, 0., 1.)
        for x in [0.1, 0.1, 0.825, 0.85, 0.875]:
            h.fill(x)
        
        self.assertEqual(h.getBinMeanX(0), 0.1)
        self.assertEqual(h.getBinMeanX(4), 0.85)

    def test_io(self):
        ho = Hist1D(5, 0., 1.)
        for x in [0.1, 0.1, 0.825, 0.85, 0.875]:
            ho.fill(x)

        ho.save('hist1d.npz')
        hi = Hist1D()
        hi.load('hist1d.npz')

        self.assertTrue(np.array_equal(ho.xedges, hi.xedges))
        self.assertTrue(np.array_equal(ho.data_n, hi.data_n))
        self.assertTrue(np.array_equal(ho.data_x, hi.data_x))
        self.assertTrue(np.array_equal(ho.data_w, hi.data_w))
        self.assertTrue(np.array_equal(ho.data_w2, hi.data_w2))

        call('rm -f hist1d.npz'.split())

        
class Hist2DTestCase(unittest.TestCase):
    """Tests for `Hist2D`."""

    def test_init(self):
        h = Hist2D(5,  0., 1.,
                   10, 0., 10.)
        self.assertEqual(h.nxbins, 5)
        self.assertEqual(h.xmin, 0.)
        self.assertEqual(h.xmax, 1.)
        self.assertEqual(h.nybins, 10)
        self.assertEqual(h.ymin, 0.)
        self.assertEqual(h.ymax, 10.)
        self.assertTrue(np.array_equal(h.xedges, np.linspace(0., 1., 6)))
        self.assertTrue(np.array_equal(h.yedges, np.linspace(0., 10., 11)))

    def test_get_bins(self):
        h = Hist2D(5,  0., 1.,
                   10, 0., 10.)
        
        self.assertEqual(h.getXBin(0.1), 0)
        self.assertEqual(h.getXBin(0.2), 1)
        self.assertEqual(h.getXBin(10.), -1)
        self.assertEqual(h.getXBin(-1.), -1)

        self.assertEqual(h.getYBin(0.1), 0)
        self.assertEqual(h.getYBin(0.2), 0)
        self.assertEqual(h.getYBin(2.0), 2)
        self.assertEqual(h.getYBin(9.9), 9)
        self.assertEqual(h.getYBin(10.), -1)

        self.assertEqual(h.getBin(0.1, 0.2), 0)
        self.assertEqual(h.getBin(0.1, 9.9), 45)

    def test_fill(self):
        h = Hist2D(5,  0., 1.,
                   10, 0., 10.)

        for x, y in zip([0.1, 0.1, 0.7, 0.72], [1., 1.5, 6.5, 6.5]):
            h.fill(x, y)

        self.assertEqual(h.getBinEntries(0,1), 2)
        self.assertEqual(h.getBinMeanX(0,1), 0.1)
        self.assertEqual(h.getBinMeanY(0,1), 1.25)
        self.assertEqual(h.getBinErr(0,1), 0.)

        self.assertEqual(h.getBinEntries(3,6), 2)
        self.assertEqual(h.getBinMeanX(3,6), 0.71)
        self.assertEqual(h.getBinMeanY(3,6), 6.5)
        self.assertEqual(h.getBinErr(3,6), 0.)

        # Test handling of overflow data (they are not stored)
        h.fill(1., 10.)
        self.assertEqual(h.getBinEntries(4,9), 0)

        h.fill(-1., -1.)
        self.assertEqual(h.getBinEntries(0,0), 0)

        h.fill(-1., 0.5)
        self.assertEqual(h.getBinEntries(0,0), 0)

    def test_2D(self):
        # Test 2D array representations of the data in the histogram
        h = Hist2D(5,  0., 1.,
                   10, 0., 10.)

        array = np.zeros((5,10), dtype=float)

        # Note that numpy uses column-major ordering and we're using row-major
        # ordering
        self.assertTrue(np.array_equal(h.getValue(), array.T))

        for x, y in zip([0.1, 0.1, 0.7, 0.72], [1., 1.5, 6.5, 6.5]):
            h.fill(x, y)

        array[0,1] = 2.
        array[3,6] = 2.

        self.assertTrue(np.array_equal(h.getValue(), array.T))

    def test_io(self):
        ho = Hist2D(5,  0., 1.,
                    10, 0., 10.)

        for x, y in zip([0.1, 0.1, 0.7, 0.72], [1., 1.5, 6.5, 6.5]):
            ho.fill(x, y)

        ho.save('hist2d.npz')
        hi = Hist2D()
        hi.load('hist2d.npz')

        self.assertTrue(np.array_equal(ho.xedges, hi.xedges))
        self.assertTrue(np.array_equal(ho.yedges, hi.yedges))
        self.assertTrue(np.array_equal(ho.data_n, hi.data_n))
        self.assertTrue(np.array_equal(ho.data_x, hi.data_x))
        self.assertTrue(np.array_equal(ho.data_y, hi.data_y))
        self.assertTrue(np.array_equal(ho.data_w, hi.data_w))
        self.assertTrue(np.array_equal(ho.data_w2, hi.data_w2))

        call('rm -f hist2d.npz'.split())

if __name__ == '__main__':
    unittest.main()
