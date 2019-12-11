import numpy as np
from matplotlib import pyplot as plt
from scipy.special import jv,kn,kv
from scipy.optimize import fsolve

from ..mode_basis import ModeBasis

def eigenvalue_equation(u, m, V):
	'''Evaluates the eigenvalue equation for a circular step-index fiber.

	Parameters
	----------
	u : scalar
		The normalized propagation constant.
	m : int
		The azimuthal order
	V : scalar
		The normalized frequency parameter of the fiber.
	
	Returns
	-------
	scalar
		The eigenvalue equation value
	'''
	w = np.sqrt(V**2 - u**2)
	return jv(m, u)/(u * jv(m+1, u)) - kn(m, w)/(w * kn(m+1, w))

def find_branch_cuts(m, V):
	'''Find all the solutions for the eigenvalue function.

	Parameters
	----------
	m : int
		The azimuthal order
	V : scalar
		The normalized frequency parameter of the fiber.
	
	Returns
	-------
	Tuple
		A tuple containing the solutions of the eigenvalue function. If no solutions were found returns None.
	'''
	# Make an initial rough grid
	num_steps = 501
	theta = np.linspace(np.pi * 9999/20000, 0.001 * np.pi, num_steps)
	u = V * np.cos(theta)

	# Find the position where it goes through zero
	diff = eigenvalue_equation(u, m, V)
	fu = np.diff(np.sign(diff)) < 0
	ind = np.where(abs(fu - 1) <= 0.01)[0]

	if len(ind) > 0:
		# Refine the zero with a rootfinding algorithm
		u0 = fsolve(eigenvalue_equation, u[ind], args=(m, V))
		w0 = np.sqrt(V**2 - u0**2)
		return u0, w0
	else:
		return None
	

def LP_radial(m, u, w, r):
	'''Evaluates the radial profile of the LP modes.

	Parameters
	----------
	m : int
		The azimuthal order
	u : scalar
		The normalized inner propagation constant.
	w : scalar
		The normalized outer propagation constant.
	r : array_like
		The radial coordinates on which to evaluate the bessel modes.
	
	Returns
	-------
	array_like
		An array that contains the radial profile.
	'''
	# The scaling factor for the continuity condition
	scaling_factor = jv(m,u)/kn(m, w)

	# Find the grid inside and outside the core radius
	mask = r < 1

	# Evaluate the radial mode profile
	mode_field = np.zeros_like(r)
	mode_field[mask] = jv(m, u * r[mask])
	mode_field[~mask] = scaling_factor * kn(m, w * r[~mask])

	return mode_field

def LP_azimuthal(m, theta):
	'''Evaluates the azimuthal profile of the LP modes.

	Parameters
	----------
	m : int
		The azimuthal order
	theta : array_like
		The azimuthal coordinates on which to evaluate the cosine and sine modes.
	
	Returns
	-------
	array_like
		An array that contains the azimuthal profile.
	'''
	if m >= 0:
		return np.cos(m * theta)
	else:
		return np.sin(m * theta)

def make_LP_modes(grid, V_number, core_radius, mode_cutoff=None):
	'''Make a ModeBasis out of the guided modes that are supported by the fiber parameters.

	Parameters
	----------
	grid : Grid
		The grid on which to calculate the mode basis.
	V_number : scalar
		The normalized frequency parameter of the fiber.
	core_radius : scalar
		The core radius of a step-index fiber.
	mode_cutoff : int
		The number of modes to find.
	Returns
	-------
	ModeBasis
		An ModeBasis containing all supported LP modes.
	'''
	finding_new_modes = True
	m = 0
	num_modes = 0

	# scaled grid
	R, Theta = grid.scaled(1/core_radius).as_('polar').coords
	modes = []
	while finding_new_modes:
		
		solutions = find_branch_cuts(m, V_number)
		if solutions is not None:
			for ui, wi in zip(solutions[0], solutions[1]):
				
				radial_prodile = LP_radial(m, ui, wi, R)

				ms = [m,-m] if m > 0 else [m,]
				for mi in ms:
					azimutal_profile = LP_azimuthal(mi, Theta)
					mode_profile = radial_prodile * azimutal_profile

					# Normalize the mode numerically as there is no analytical normalization
					norm = np.sqrt( np.sum(mode_profile * mode_profile.conj() * grid.weights) )
					mode_profile /= norm
					modes.append(mode_profile)
					num_modes += 1

			m += 1
		else:
			finding_new_modes = False
	
	return ModeBasis(modes, grid)