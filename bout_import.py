"""Paraview reader for BOUT++ data files

===============
2015 Peter Hill
"""

import os.path
import numpy as np
import warnings

from boututils.datafile import DataFile
import boutdata

from vtk import vtkPoints
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa
from paraview import numpy_support

import upscale

def set_output_timesteps(pipeline, timesteps):
    """Helper routine to set timestep info
    """

    executive = pipeline.GetExecutive()
    out_info = executive.GetOutputInformation(0)

    out_info.Remove(executive.TIME_STEPS())

    for timestep in timesteps:
        out_info.Append(executive.TIME_STEPS(), timestep)

    out_info.Remove(executive.TIME_RANGE())
    out_info.Append(executive.TIME_RANGE(), timesteps[0])
    out_info.Append(executive.TIME_RANGE(), timesteps[-1])

def get_update_timestep(pipeline):
    """Returns the requested time value, or None if not present
    """
    executive = pipeline.GetExecutive()
    out_info = executive.GetOutputInformation(0)
    if not out_info.Has(executive.UPDATE_TIME_STEP()):
        return None
    return out_info.Get(executive.UPDATE_TIME_STEP())

def set_output_extent(pipeline, nx, ny, nz):
    """Helper routine to set extent info
    """

    executive = pipeline.GetExecutive()
    out_info = executive.GetOutputInformation(0)

    out_info.Set(executive.WHOLE_EXTENT(), 0, nx-1, 0, ny-1, 0, nz-1)

def request_info(pipeline, filename, upscale_factor=1):
    """RequestInformation script
    """

    directory = os.path.dirname(filename)

    # Read timesteps from first file
    with DataFile(filename) as f:
        # Get timesteps
        t_array = f.read("t_array")
        if t_array is None:
            t_array = np.zeros(1)

        # Get list of variables with at least 2 dimensions
        var_list = [var for var in f.list() if f.ndims(var) >= 2]

    # Tell paraview about the timesteps
    set_output_timesteps(pipeline, t_array)

    # Get dataset sizes
    dx = boutdata.collect("dx", path=directory, info=False)
    nx, ny = dx.shape
    nz = boutdata.collect("MZ", path=directory, info=False)
    nz = nz - 1

    ny = upscale_factor*ny

    set_output_extent(pipeline, nx, ny, nz)

    return directory

def request_data(pipeline, filename, gridfile=None, upscale_factor=1):
    """Read the datasets from the output files
    """

    # upscale_factor must be integer and >= 1
    if not isinstance(upscale_factor, int):
        raise TypeError("upscale_factor must be integer (upscale_factor = {})"
                        .format(upscale_factor))
    elif upscale_factor <= 0:
        raise TypeError("upscale_factor must be larger than 1 (upscale_factor = {})"
                        .format(upscale_factor))

    directory = pipeline.directory

    # Collect dx, dy, dz
    dx = boutdata.collect("dx", path=directory, info=False)
    dy = boutdata.collect("dy", path=directory, info=False)
    dz = boutdata.collect("dz", path=directory, info=False)

    # Grids
    nx, ny = dx.shape
    nz = boutdata.collect("MZ", path=directory, info=False)
    nz = nz - 1
    x = np.linspace(0, nx*dx[0,0], nx)
    y = np.linspace(0, ny*dy[0,0], upscale_factor*ny, endpoint=False)
    z = np.linspace(0, nz*dz, nz, endpoint=False)

    X,Y,Z = np.meshgrid(x,y,z, indexing='ij')

    # Make coordinates array
    coordinates = algs.make_vector(X.T.flatten(), Y.T.flatten(), Z.T.flatten())
    points = vtkPoints()
    points.SetData(dsa.numpyTovtkDataArray(coordinates, 'Points'))

    # VTK output object
    sgo = pipeline.GetStructuredGridOutput()

    # Set size of output object
    sgo.SetDimensions(nx, upscale_factor*ny, nz)
    sgo.SetExtent(0, nx-1, 0, upscale_factor*ny-1, 0, nz-1)

    # Add coordinate points
    sgo.SetPoints(points)

    with DataFile(filename) as f:
        # Get list of variables with at least 2 dimensions
        var_list = [var for var in f.list() if f.ndims(var) > 2]
        # Get timesteps
        t_array = f.read("t_array")
        if t_array is None:
            t_array = np.zeros(1)

    # Get the requested timestep
    req_time = get_update_timestep(pipeline)
    timestep = (np.abs(t_array-req_time)).argmin()

    output = pipeline.GetOutput()
    output.GetInformation().Set(output.DATA_TIME_STEP(), req_time)

    # Add the variables to the output
    add_fields_to_sgo(sgo, var_list, directory, timestep, gridfile, upscale_factor)

    return pipeline

def add_fields_to_sgo(sgo, var_list, directory, timestep=None, gridfile=None, upscale_factor=1):
    """Add the fields with more than 2 dimensions to the grid output
    """

    if timestep is None:
        timestep = 0

    for var in var_list:
        # Read data
        bout_data = boutdata.collect(var, tind=timestep, path=directory, info=False)

        if gridfile is not None and upscale_factor > 1:
            bout_data = upscale.upscale(np.squeeze(bout_data), gridfile, upscale_factor)
        elif gridfile is None and upscale_factor > 1:
            warnings.warn("upscale_factor is {}, but no gridfile provided".format(upscale_factor),
                          RuntimeWarning)

        # Convert from numpy to vtk
        vtk_data = numpy_support.numpy_to_vtk(bout_data.T.flatten().copy(), deep=1)
        # Set field name
        vtk_data.SetName(var)

        # Add array to sgo
        sgo.GetPointData().SetScalars(vtk_data)
