""" Module to convert redshift into comoving distance given cosmology

AstroPy has internally defined cosmologies based on recent CMB missions
(see http://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies).
We'll use those parameters here to define comoving distance, but with
two changes:

1) Define H0 = 100h km/s/Mpc; astropy default uses the measured value of H0.
2) Allow for the possibile definition of nonzero curvature (not by default).
"""
import os
import errno
import numpy
from astropy import cosmology, units
from scipy import interpolate

class Cosmology():
    """ Class to manage cosmological parameters and convert redshift z into
        comoving distance using linear interpolation technique. """
    def __init__(self):
        """ Initialize a cosmological model. Note that H0 = 100 h km/s/Mpc
        is used. For now the cosmological parameters measured by Planck
        (P.A.R. Ade et al., Paper XIII, A&A 594:A13, 2016) are used.
        """
        try:
            os.makedirs("model_cache")
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        self.model = 0
        self.__comoving_table = 0
        self.set_model()

    def __set_comoving_table(self):
        """ Return the redshift-comoving distance table from z_min to z_max.
        If there already exists a table, read from table and return false. Else
        create a table and return true.
        Inputs:
        + z_min: int, float
            Minimum redshift.
        + z_max: int, float
            Maximum redshift.
        + step: int, float
            Redshift step.
        Outputs:
        + table: ndarray
            Return 2-d array where the zeroth index stores redshift and the
            first index stores comoving distance.
        """
        # default table parameters
        z_min = 0.
        z_max = 3.0
        step = 0.00001
        # if cache file does not exist, create a table
        isfile = os.path.isfile(self.__cache_str())
        if not isfile:
            num_entries = int(numpy.ceil((z_max-z_min)/step))
            table_z = numpy.linspace(z_min, z_max, num_entries)
            table_r = self.model.comoving_distance(table_z)
            self.__comoving_table = numpy.array([table_z, table_r]).T
            numpy.savetxt(self.__cache_str(), self.__comoving_table,
                          header="z    r(Mpc/h)")  # cache
        else:
            self.__comoving_table = numpy.genfromtxt(self.__cache_str())
        return isfile

    def __cache_str(self):
        """ Get path to cache file given cosmology input"""
        model_str = ("Hubble{}_Baryon{}_DarkMatter{}_DarkEnergy{}"
                     "_Tcmb{}_Neff{}_Mnu{},{},{}").format(
                         self.model.H0.value,
                         self.model.Om0,
                         self.model.Ob0,
                         self.model.Ode0,
                         self.model.Tcmb0,
                         self.model.Neff,
                         *self.model.m_nu)
        fname = "model_cache/{}.txt".format(model_str)
        return fname

    def get_comoving_table(self):
        """ Return a copy of the comoving distance table """
        return numpy.copy(self.__comoving_table)

    def set_model(self, H0=100, Om0=0.307, Ob0=0.0486,
                  Ode0=0.693, Tcmb0=2.725,
                  Neff=3.05, m_nu=[0., 0., 0.06]):
        """ Set cosmology parameters
        Inputs:
        + H0: int, float (default=100)
            Hubble constant at present (z=0). Unit in km/(Mpc*s)
        + Om0: int, float (default=0.307)
            Matter density/critical density at z=0. Must be greater than 0.
        + Omb0: int, float (default=0.0486)
            Baryonic matter density/critical density at present (z=0). Must be
            greater than 0.
        + Ode0: int, float (default=0.694)
            Dark energy density/critical density at z=0
        + Tcmb0: int, float (default=2.725)
            Temperature of CMB at present (z=0). Must be greater than 0.
        + Neff: int, float (default=3.05)
            Effective number of Neutrino species.
        + m_nu: int, float, array_like (default=[0., 0., 0.06])
            Mass of each neutrino species. If this is a scalar quantity, then
            all neutrino species are assumed to have that mass. Otherwise, the
            mass of each species. The actual number of neutrino species
            (and hence the number of elements of m_nu if it is not scalar) must
            be the floor of Neff. Typically this means you should provide three
            neutrino masses unless you are considering something like a sterile
            neutrino.
        """
        self.model = cosmology.LambdaCDM(H0=H0*units.km/(units.Mpc*units.s),
                                         Om0=Om0,
                                         Ob0=Ob0,
                                         Ode0=Ode0,
                                         Tcmb0=Tcmb0,
                                         Neff=Neff,
                                         m_nu=numpy.asarray(m_nu)*units.eV)
        # Initialize array of redshift and comoving distance
        self.__set_comoving_table()

    def z2r(self, z):
        """Convert redshift z into comoving distance r by linear interpolating
        from comoving distance table.
        Inputs:
        + z: int, float
            Redshift. Must be greater than or equal to 0, and less than or equal
            to 3.
        Outputs:
        + comoving distance: float
            Return comoving distance given redshift and set cosmology.
        """
        return interpolate.PchipInterpolator(*self.__comoving_table.T)(z)
