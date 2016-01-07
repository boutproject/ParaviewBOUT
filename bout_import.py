from boututils.datafile import DataFile
import boutdata
from glob import glob
import os.path
import numpy as np
from vtk import vtkPoints, vtkStructuredGrid, vtkDoubleArray
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa
from paraview import numpy_support

def get_bout_datafiles(directory):
    # Read bout datafiles
    return sorted(glob(os.path.join(directory, "BOUT.dmp.*nc")))

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

    # out_info.Remove(executive.WHOLE_EXTENT())
    out_info.Set(executive.WHOLE_EXTENT(), 0, nx-1, 0, ny-1, 0, nz-1)
    # print(out_info)

def request_info_script(pipeline, directory="/home/peter/Codes/BOUT-dev/examples/stellarator/n16_noyend"):
    """RequestInformation script
    """

    import PyQt4.QtGui
    filename = PyQt4.QtGui.QFileDialog.getOpenFileName(None, "Select a file...")

    # print("Requesting info")
    # Get list of output files
    file_list = get_bout_datafiles(directory)

    # Read timesteps from first file
    # print("Variables in file:")
    with DataFile(file_list[0]) as f:
        # Get timesteps
        t_array = f.read("t_array")
        if t_array is None:
            t_array = np.zeros(1)

        # Get list of variables with at least 2 dimensions
        var_list = [var for var in f.list() if f.ndims(var) >= 2]
        # print(var_list)

    # Tell paraview about the timesteps
    set_output_timesteps(pipeline, t_array)
    # print("Timesteps:")
    # print(t_array)

    # Get dataset sizes
    dx = boutdata.collect("dx", path=directory, info=False)
    nx, ny = dx.shape
    nz = boutdata.collect("MZ", path=directory, info=False)
    nz = nz - 1

    set_output_extent(pipeline, nx, ny, nz)

def get_update_timestep(pipeline):
    """Returns the requested time value, or None if not present
    """
    executive = pipeline.GetExecutive()
    out_info = executive.GetOutputInformation(0)
    if out_info.Has(executive.UPDATE_TIME_STEP()):
        return out_info.Get(executive.UPDATE_TIME_STEP())
    else:
        return None

def import_bout(pipeline, directory="/home/peter/Codes/BOUT-dev/examples/stellarator/n16_noyend"):

    file_list = get_bout_datafiles(directory)
    # print(file_list)
    
    # Collect dx, dy, dz
    dx = boutdata.collect("dx", path=directory, info=False)
    dy = boutdata.collect("dy", path=directory, info=False)
    dz = boutdata.collect("dz", path=directory, info=False)

    # Grids
    nx, ny = dx.shape
    nz = boutdata.collect("MZ", path=directory, info=False)
    nz = nz - 1
    x = np.linspace(0, nx*dx[0,0], nx)
    y = np.linspace(0, ny*dy[0,0], ny)
    z = np.linspace(0, nz*dz, nz)

    X,Y,Z = np.meshgrid(x,y,z, indexing='ij')

    # Make coordinates array
    coordinates = algs.make_vector(X.T.flatten(), Y.T.flatten(), Z.T.flatten())
    points = vtkPoints()
    points.SetData(dsa.numpyTovtkDataArray(coordinates, 'Points'))

    # VTK output object
    sgo = pipeline.GetStructuredGridOutput()
    # print(sgo)

    # Not sure if needed - taken from gs2 script
    # if not hasattr(sgo, "GetProducerPort"):
    #     print("Using vtkStructuredGrid")
    #     sgo = vtkStructuredGrid()

    # Set size of output object
    sgo.SetDimensions(nx, ny, nz)
    sgo.SetExtent(0, nx-1, 0, ny-1, 0, nz-1)

    # Add coordinate points
    sgo.SetPoints(points)

    with DataFile(file_list[0]) as f:
        # Get list of variables with at least 2 dimensions
        var_list = [var for var in f.list() if f.ndims(var) > 2]
        # Get timesteps
        t_array = f.read("t_array")
        if t_array is None:
            t_array = np.zeros(1)

    # Get the requested timestep
    req_time = get_update_timestep(pipeline)
    timestep = (np.abs(t_array-req_time)).argmin()

    # Add the variables to the output
    add_fields_to_sgo(sgo, var_list, directory, timestep=timestep)

    # print(sgo)
    # print(sgo.GetPoints())

    # pipeline.SetStructuredGridOutput(sgo)
    
    return pipeline

def add_fields_to_sgo(sgo, var_list, directory, timestep=None):
    """Add the fields with more than 2 dimensions to the grid output
    """

    if timestep is None:
        timestep = 0

    for var in var_list:
        # VTK array to hold data
        # vtk_data = vtkDoubleArray()

        # Read data
        bout_data = boutdata.collect(var, tind=timestep, path=directory, info=False)
        # print(var)
        # print(bout_data.shape)

        # Convert from numpy to vtk
        vtk_data = numpy_support.numpy_to_vtk(np.squeeze(bout_data).T.flatten().copy(), deep=1)
        # Set field name
        vtk_data.SetName(var)

        # Add array to sgo
        sgo.GetPointData().SetScalars(vtk_data)
