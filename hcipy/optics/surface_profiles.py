import numpy as np
from ..field import Field

def spherical_surface_sag(radius_of_curvature):
	'''Makes a Field generator for the surface sag of an even aspherical surface.

	Parameters
	----------
	radius_of_curvature : scalar
		The radius of curvature of the surface.

	Returns
	-------
	Field generator
		This function can be evaluated on a grid to get the sag profile.
	'''
	return conical_surface_sag(radius_of_curvature, conic_constant=0)

def parabolic_surface_sag(radius_of_curvature):
	'''Makes a Field generator for the surface sag of an even aspherical surface.

	Parameters
	----------
	radius_of_curvature : scalar
		The radius of curvature of the surface.

	Returns
	-------
	Field generator
		This function can be evaluated on a grid to get the sag profile.
	'''
	return conical_surface_sag(radius_of_curvature, conic_constant=-1)

def conical_surface_sag(radius_of_curvature, conic_constant=0):
	r'''Makes a Field generator for the surface sag of a conical surface.

	The surface profile is defined as:

	.. math:: z = \frac{cr^2}{1 + \sqrt{1-\left(1+k\right)c^2r^2}}

	with `z` the surface sag, `c` the curvature and `k` the conic constant.

	Parameters
	----------
	radius_of_curvature : scalar
		The radius of curvature of the surface.
	conic_constant : scalar
		The conic constant of the surface

	Returns
	-------
	Field generator
		This function can be evaluated on a grid to get the sag profile.
	'''
	def func(grid):
		if grid.is_('cartesian'):
			if grid.is_separated:
				if grid.ndim == 2:
					#x, y = grid.separated_coords
					#r = np.hypot(x, y)
					r = grid.as_('polar').r
				else:
					r = grid.separated_coords
			else:
				if grid.ndim == 2:
					x, y = grid.coords
					r = np.hypot(x, y)
				else:
					r = x
		else:
			r = grid.r
		curvature = 1 / radius_of_curvature
		alpha = (1 + conic_constant) * curvature**2 * r**2
		
		mask = r < abs(radius_of_curvature)
		sag = grid.zeros()
		sag[mask] = r[mask]**2 / (radius_of_curvature * (1 + np.sqrt(1 - alpha[mask])))

		return Field(sag, grid)

	return func

def even_aspheric_surface_sag(radius_of_curvature, conic_constant=0, aspheric_coefficients=None):
	r'''Makes a Field generator for the surface sag of an even aspherical surface.

	The surface profile is defined as:

	.. math:: z = \frac{cr^2}{1 + \sqrt{1-\left(1+k\right)c^2r^2}} + \sum_i=0 a_i r^{2i+4}

	With `z` the surface sag, `c` the curvature, `k` the conic constant and  :math:`a_i` the even aspheric coefficients.

	It is important to note that this definition deviates from the Zemax definition of an even aspheric surface.
	In Zemax the 2nd order term is also included in the expansion,
	which is unnessary because the conic surface itself already accounts for the 2nd order term.

	Parameters
	----------
	radius_of_curvature : scalar
		The radius of curvature of the surface.
	conic_constant : scalar
		The conic constant of the surface
	aspheric_coefficients : array_like
		Contains the high-order even aspheric coefficients.

	Returns
	-------
	Field generator
		This function can be evaluated on a grid to get the sag profile.
	'''
	if aspheric_coefficients is None:
		aspheric_coefficients = []

	def func(grid):
		'''
		if grid.is_('cartesian'):
			if grid.is_separated:
				if grid.ndim == 2:
					x, y = grid.separated_coords
					r = np.hypot(x, y)
				else:
					r = grid.separated_coords[0]
			else:
				if grid.ndim == 2:
					x, y = grid.coords
					r = np.hypot(x, y)
				else:
					r = grid.coords[0]
		else:
			r = grid.r
		'''
		x, y = grid.coords
		r = np.hypot(x, y)
		#print(r.shape)
		
		# Start with a conic surface
		curvature = 1 / radius_of_curvature
		alpha = (1 + conic_constant) * curvature**2 * r**2
		sag = r**2 / (radius_of_curvature * (1 + np.sqrt(1 - alpha)))

		# Add aspheric coefficients
		# Only use the even modes and start at 4, because 0 is piston and 2 is the conic surface
		for ai, coef in enumerate(aspheric_coefficients):
			power_index = 4 + ai * 2
			sag += coef/1e-6**(power_index-1) * r**power_index
		return Field(sag, grid)

	return func
