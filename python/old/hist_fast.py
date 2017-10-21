# -*- coding: utf-8 -*-
"""Fast rectangular histograms.

This module implements fast histogramming in 1D and 2D with even bins.
The histograms can be filled on the fly, allowing single passes through
large datasets with very low memory overhead.
"""

import numpy as np

class Hist1D:
    """A fast 1D histogram class that can be filled on the fly.

    This class is used to produce 1D weighted histograms which can be 
    filled on the fly given an ordinate and a weight (x, w). One unusual
    feature is that it tracks the mean value of the ordinate x pushed 
    into each bin.

    Attributes:
        nxbins (:obj:`int`): number of bins.
        xmin (:obj:`float`): minimum of ordinate range.
        xmax (:obj:`float`): maximum of ordinate range.
        data_n (:obj:`numpy.array` of :obj:`float`): bin counts.
        data_w (:obj:`numpy.array` of :obj:`float`): sum of weights.
        data_w2 (:obj:`numpy.array` of :obj:`float`): quad sum of weights.
        data_x (:obj:`numpy.array` of :obj:`float`): sum of ordinates.
        xedges (:obj:`numpy.array` of :obj:`float`): array of bin edges.
    """

    def __init__(self, nxbins=10, xmin=0, xmax=1):
        """Initialize histogram with number of bins and x range.

        Note:
            Only equal binning is currently supported.

        Args:
            nxbins (:obj:`int`): number of bins.
            xmin (:obj:`float`): minimum of ordinate range.
            xmax (:obj:`float`): maximum of ordinate range.
        """
        # Bin number and limits
        self.nxbins = nxbins
        self.xmin = xmin
        self.xmax = xmax

        self.data_n  = np.zeros(nxbins, dtype=float)
        self.data_w  = np.zeros(nxbins, dtype=float)
        self.data_w2 = np.zeros(nxbins, dtype=float)
        self.data_x  = np.zeros(nxbins, dtype=float)

        self.xedges   = np.linspace(xmin, xmax, nxbins+1)

    def fill(self, x, w=1.):
        """Fill histogram with ordinate x and weight w.

        Note:
            The histogram does not keep overflow bins; x outside the
            histogram range will be thrown out.

        Args:
            x (:obj:`float`): ordinate used for binning.
            w (:obj:`float`): ordinate weight.
        """
        i = int(self.nxbins * (x-self.xmin)/(self.xmax-self.xmin))
        if i < 0 or i >= self.nxbins:
            return

        self.data_n[i]  += 1;
        self.data_w[i]  += w;
        self.data_w2[i] += w*w;
        self.data_x[i]  += x;

    def getBinEntries(self, i):
        """Get the number of entries in the ith bin.

        Args:
            i (:obj:`int`): bin index (0..n-1)
        """
        return self.data_n[i]

    def getBinValue(self, i):
        """Get the weighted sum in the ith bin.

        Args:
            i (:obj:`int`): bin index (0..n-1)
        """
        return self.data_w[i]

    def getBinErr(self, i):
        """Get the uncertainty in the weighted sum of the ith bin.

        Args:
            i (:obj:`int`): bin index (0..n-1)

        Returns:
            Sqrt of variance in bin i if n[i]>0, and zero otherwise.
        """
        if self.data_n[i] > 1:
            return np.sqrt(1./(self.data_n[i]-1) * \
                (self.data_w2[i] - self.data_w[i]**2/self.data_n[i]))
        return 0.

    def getBinMeanX(self, i):
        """Get mean of the x ordinates used when filling each bin.

        Args:
            i (:obj:`int`): bin index (0..n-1)

        Returns:
            Mean x of bin i if n[i]>0, and bin center otherwise.
        """
        if self.data_n[i] > 0:
            return self.data_x[i]/self.data_n[i]
        return (self.xmax-self.xmin)/self.nxbins * \
                (0.5 + (i % self.nxbins)) + \
                self.xmin

    def save(self, outfile):
        """Save histogram to a NumPy compressed binary file.
        
        Args:
            outfile (:obj:`str`): name of output file (.npz extension).
        """
        np.savez_compressed(outfile, xedges=self.xedges, 
                                     data_n=self.data_n,
                                     data_w=self.data_w,
                                     data_w2=self.data_w2,
                                     data_x=self.data_x)

    def load(self, infile):
        """Load histogram from a compressed binary file.
        
        Args:
            infile (:obj:`str`): name of NPZ format input file.
        """
        f = np.load(infile)

        self.data_n  = f['data_n']
        self.data_w  = f['data_w']
        self.data_w2 = f['data_w2']
        self.data_x  = f['data_x']

        self.xedges  = f['xedges']
        self.xmin, self.xmax = self.xedges[0], self.xedges[-1]

        self.nxbins = len(self.data_n)


class Hist2D:
    """A fast 2D histogram class that can be filled on the fly.

    This class is used to produce 2D weighted histograms which can be 
    filled on the fly given ordinates and a weight (x,y, w). One unusual
    feature is that it tracks the mean value of the ordinates x,y pushed 
    into each bin.

    Attributes:
        nxbins (:obj:`int`): number of x bins.
        xmin (:obj:`float`): minimum of ordinate range.
        xmax (:obj:`float`): maximum of ordinate range.
        nybins (:obj:`int`): number of y bins.
        ymin (:obj:`float`): minimum of ordinate range.
        ymax (:obj:`float`): maximum of ordinate range.
        data_n (:obj:`numpy.array` of :obj:`float`): bin counts.
        data_w (:obj:`numpy.array` of :obj:`float`): sum of weights.
        data_w2 (:obj:`numpy.array` of :obj:`float`): quad sum of weights.
        data_x (:obj:`numpy.array` of :obj:`float`): sum of ordinates x.
        data_y (:obj:`numpy.array` of :obj:`float`): sum of ordinates y.
        xedges (:obj:`numpy.array` of :obj:`float`): array of bin edges.
        yedges (:obj:`numpy.array` of :obj:`float`): array of bin edges.
    """

    def __init__(self, nxbins=10, xmin=0, xmax=1, nybins=10, ymin=0, ymax=1):
        """Initialize histogram with number of bins and x-y range.

        Note:
            Only equal binning is currently supported.

        Args:
            nxbins (:obj:`int`): number of x bins.
            xmin (:obj:`float`): minimum of ordinate range.
            xmax (:obj:`float`): maximum of ordinate range.
            nybins (:obj:`int`): number of y bins.
            ymin (:obj:`float`): minimum of ordinate range.
            ymax (:obj:`float`): maximum of ordinate range.
        """
        # Bin number and limits
        self.nxbins = nxbins
        self.xmin = xmin
        self.xmax = xmax
        self.nybins = nybins
        self.ymin = ymin
        self.ymax = ymax

        # Storage for data
        self.data_n  = np.zeros(nxbins*nybins, dtype=float)
        self.data_w  = np.zeros(nxbins*nybins, dtype=float)
        self.data_w2 = np.zeros(nxbins*nybins, dtype=float)
        self.data_x  = np.zeros(nxbins*nybins, dtype=float)
        self.data_y  = np.zeros(nxbins*nybins, dtype=float)

        # Bin edges
        self.xedges  = np.linspace(xmin, xmax, nxbins+1)
        self.yedges  = np.linspace(ymin, ymax, nybins+1)

    def getXBin(self, x):
        """Get the bin corresponding to ordinate x.

        Note:
            The histogram does not allow overflow bins.

        Args:
            x (:obj:`float`): ordinate used for binning.

        Returns:
            If x is in bin range, the bin number; else, -1.
        """
        i = int(self.nxbins * (x-self.xmin)/(self.xmax-self.xmin))
        if i >= 0 and i < self.nxbins:
            return i
        return -1

    def getYBin(self, y):
        """Get the bin corresponding to ordinate y.

        Note:
            The histogram does not allow overflow bins.

        Args:
            y (:obj:`float`): ordinate used for binning.

        Returns:
            If y is in bin range, the bin number; else, -1.
        """
        j = int(self.nybins * (y-self.ymin)/(self.ymax-self.ymin))
        if j >= 0 and j < self.nybins:
            return j
        return -1

    def getBin(self, x, y):
        """Get the bin corresponding to ordinates x, y.

        Note:
            The histogram does not allow overflow bins.

        Args:
            x (:obj:`float`): ordinate used for binning in x.
            y (:obj:`float`): ordinate used for binning in y.

        Returns:
            If x, y in bin range, the bin number; else, -1.
        """
        i = self.getXBin(x)
        j = self.getYBin(y)
        if i == -1 or j == -1:
            return -1
        return i + self.nxbins * j

    def fill(self, x, y, w=1.):
        """Fill histogram with ordinates x, y and weight w.

        Note:
            The histogram does not keep overflow bins; x,y outside the
            histogram range will be thrown out.

        Args:
            x (:obj:`float`): ordinate used for binning in x.
            y (:obj:`float`): ordinate used for binning in y.
            w (:obj:`float`): ordinate weight.
        """
        k = self.getBin(x, y)
        if k == -1:
            return

        self.data_n[k]  += 1;
        self.data_w[k]  += w;
        self.data_w2[k] += w*w;
        self.data_x[k]  += x;
        self.data_y[k]  += y;

    def getBinEntries(self, i, j):
        k = i + self.nxbins*j
        return self.data_n[k]

    def getBinValue(self, i, j):
        k = i + self.nxbins*j
        return self.data_w[k]

    def getBinErr(self, i, j):
        k = i + self.nxbins*j
        if self.data_n[k] > 1:
            return np.sqrt(1./(self.data_n[k]-1) * \
                (self.data_w2[k] - self.data_w[k]**2/self.data_n[k]))
        return 0.

    def getBinMeanX(self, i, j):
        k = i + self.nxbins*j
        if self.data_n[k] > 0:
            return self.data_x[k] / self.data_n[k]
        return (self.xmax-self.xmin)/self.nxbins * (0.5 + (k % self.nxbins)) + \
                self.xmin

    def getBinMeanY(self, i, j):
        k = i + self.nxbins*j
        if self.data_n[k] > 0:
            return self.data_y[k] / self.data_n[k]
        return (self.ymax-self.ymin)/self.nybins * (0.5 + (k % self.nybins)) + \
                self.ymin

    def getEntries(self):
        # Reshape to 2D, noting that numpy using column-major ordering
        return np.flipud(np.reshape(self.data_n, (self.nybins, self.nxbins)))

    def getValue(self):
        # Reshape to 2D, noting that numpy using column-major ordering
        return np.reshape(self.data_w, (self.nybins, self.nxbins))

    def getErr(self):
        err = np.zeros(self.nxbins*self.nybins, dtype=float)
        non0 = self.data_n > 1
        err[non0] = np.sqrt(1./(self.data_n[non0]-1.) * \
            (self.data_w2[non0] - self.data_w[non0]**2/self.data_n[non0]))
        # Reshape to 2D, noting that numpy using column-major ordering
        return np.reshape(err, (self.nxbins, self.nybins))

    def save(self, outfile):
        """Save histogram to a NumPy compressed binary file.
        
        Args:
            outfile (:obj:`str`): name of output file (.npz extension).
        """
        np.savez_compressed(outfile, xedges=self.xedges, 
                                     yedges=self.yedges,
                                     data_n=self.data_n,
                                     data_w=self.data_w,
                                     data_w2=self.data_w2,
                                     data_x=self.data_x,
                                     data_y=self.data_y)

    def load(self, infile):
        """Load histogram from a compressed binary file.
        
        Args:
            infile (:obj:`str`): name of NPZ format input file.
        """
        f = np.load(infile)

        self.data_n  = f['data_n']
        self.data_w  = f['data_w']
        self.data_w2 = f['data_w2']
        self.data_x  = f['data_x']
        self.data_y  = f['data_y']

        self.xedges  = f['xedges']
        self.xmin, self.xmax = self.xedges[0], self.xedges[-1]
        self.nxbins = len(self.data_x)

        self.yedges  = f['yedges']
        self.ymin, self.ymax = self.yedges[0], self.yedges[-1]
        self.nybins = len(self.data_y)
