import boutdata
from boututils.datafile import DataFile
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import griddata
from vtk import vtkPoints
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa
from paraview import numpy_support


def upscale(field, gridfile, upscale_factor=4):
    """Increase the resolution in y of field along the FCI maps from
    gridfile.
    """

    # Read the FCI maps from the gridfile
    with DataFile(gridfile) as grid:
        xt_prime = grid.read("forward_xt_prime")
        zt_prime = grid.read("forward_zt_prime")

    # The field should be the same shape as the grid
    if field.shape != xt_prime.shape:
        try:
            field = field.reshape(xt_prime.T.shape).T
        except ValueError:
            raise ValueError("Field, {}, must be same shape as grid, {}"
                             .format(field.shape, xt_prime.shape))

    # Get the shape of the grid
    nx, ny, nz = xt_prime.shape
    index_coords = np.mgrid[0:nx, 0:ny, 0:nz]

    # We use the forward maps, so get the y-index of the *next* y-slice
    yup_3d = index_coords[1,...] + 1
    yup_3d[:,-1,:] = 0

    # Index space coordinates of the field line end points
    end_points = np.array([xt_prime, yup_3d, zt_prime])

    # Interpolation of the field at the end points
    field_prime = map_coordinates(field, end_points)

    # This is a 4D array where the first dimension is the start/end of
    # the field line
    field_aligned = np.array([field, field_prime])

    # x, z coords at start/end of field line
    x_start_end = np.array([index_coords[0,...], xt_prime])
    z_start_end = np.array([index_coords[2,...], zt_prime])

    # Parametric points along the field line
    midpoints = np.linspace(0, 1, upscale_factor, endpoint=False)
    # Need to make this 4D as well
    new_points = np.tile(midpoints[:,np.newaxis,np.newaxis,np.newaxis], [nx, ny, nz])

    # Index space coordinates of our upscaled field
    index_4d = np.mgrid[0:upscale_factor,0:nx,0:ny,0:nz]
    hires_points = np.array([new_points, index_4d[1,...], index_4d[2,...], index_4d[3,...]])

    # Upscale the field
    hires_field = map_coordinates(field_aligned, hires_points)
    # Linearly interpolate the x, z coordinates of the field lines
    hires_x = map_coordinates(x_start_end, hires_points)
    hires_z = map_coordinates(z_start_end, hires_points)

    def twizzle(array):
        """Transpose and reshape the output of map_coordinates to
        be 3D
        """
        return array.transpose((1, 2, 0, 3)).reshape((nx, upscale_factor*ny, nz))

    # Rearrange arrays to be 3D
    hires_field = twizzle(hires_field)
    hires_x = twizzle(hires_x)
    hires_z = twizzle(hires_z)

    # Interpolate from field line sections onto grid
    hires_grid_field = np.zeros( (nx, upscale_factor*ny, nz) )
    hires_index_coords = np.mgrid[0:nx, 0:ny:1./upscale_factor, 0:nz]
    grid_points = (hires_index_coords[0,:,0,:], hires_index_coords[2,:,0,:])

    def y_first(array):
        """Put the middle index first
        """
        return array.transpose((0, 2, 1))

    # The hires data is unstructed only in (x,z), interpolate onto
    # (x,z) grid for each y-slice individually
    for k, (x_points, z_points, f_slice) in enumerate(zip(y_first(hires_x).T, y_first(hires_z).T, y_first(hires_field).T)):
        points = np.column_stack((x_points.flat, z_points.flat))
        hires_grid_field[:,k,:] = griddata(points, f_slice.flat, grid_points,
                                           method='linear', fill_value=0.0)

    return hires_grid_field
