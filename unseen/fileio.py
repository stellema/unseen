"""Functions for file I/O"""

import pdb
import os
import git
import yaml
import shutil
import zipfile
import xarray as xr
import cmdline_provenance as cmdprov

import general_utils
import spatial_selection
import time_utils
import array_handling


def open_file(infile,
              chunks='auto',
              metadata_file=None,
              variables=[],
              region=None,
              no_leap_days=False,
              time_freq=None,
              time_agg=None,
              input_freq=None,
              isel={},
              sel={},
              units={},
              ):
    """Create an xarray Dataset from an input zarr file.

    Args:
      infile (str) : Input file path
      chunks (dict) : Chunks for xarray.open_zarr 
      metadata_file (str) : YAML file specifying required file metadata changes
      variables (list) : Variables of interest
      region (str) : Spatial subset (extract this region)
      no_leap_days (bool) : Remove leap days from data
      time_freq (str) : Target temporal frequency for resampling
      time_agg (str) : Temporal aggregation method ('mean' or 'sum')
      input_freq (str) : Input time frequency for resampling (estimated if not provided) 
      isel (dict) : Selection using xarray.Dataset.isel
      sel (dict) : Selection using xarray.Dataset.sel
      units (dict) : Variable/s (keys) and desired units (values)
    """

    ds = xr.open_zarr(infile, consolidated=True, use_cftime=True, chunks=chunks)
    #if chunks:
    #    ds = ds.chunk(input_chunks)
    
    # Metadata
    if metadata_file:
        ds = fix_metadata(ds, metadata_file, variables)

    # Variable selection
    if variables:
        ds = ds[variables]

    # Spatial subsetting and aggregation
    if region:
        ds = spatial_selection.select_region(ds, region)

    # Temporal aggregation
    #with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    if no_leap_days:
        ds = ds.sel(time=~((ds['time'].dt.month == 2) & (ds['time'].dt.day == 29)))
    if time_freq:
        assert time_agg, """Provide a time_agg ('mean' or 'sum')"""
        ds = time_utils.temporal_aggregation(ds, time_freq, time_agg, input_freq=input_freq)

    # General selection/subsetting
    if isel:
        ds = ds.isel(isel)
    if sel:
        ds = ds.sel(sel)

    # Units
    for var, target_units in units.items():
        ds[var] = general_utils.convert_units(ds[var], target_units)

    assert type(ds) == xr.core.dataset.Dataset

    return ds


def open_mfforecast(infiles, **kwargs):
    """Open multi-file forecast."""

    datasets = []
    for infile in infiles:
        ds = open_file(infile, **kwargs)
        ds = array_handling.to_init_lead(ds)
        datasets.append(ds)
    ds = xr.concat(datasets, dim='init_date')

    time_values = [ds.get_index('init_date').shift(int(lead), 'D') for lead in ds['lead_time']]
    time_dimension = xr.DataArray(time_values,
                                  dims={'lead_time': ds['lead_time'],
                                        'init_date': ds['init_date']})
    ds = ds.assign_coords({'time': time_dimension})
    ds['lead_time'].attrs['units'] = 'D'

    return ds


def fix_metadata(ds, metadata_file, variables):
    """Edit the attributes of an xarray Dataset.
    
    ds (xarray Dataset or DataArray)
    metadata_file (str) : YAML file specifying required file metadata changes
    variables (list): Variables to rename (provide target name)
    """
 
    with open(metadata_file, 'r') as reader:
        metadata_dict = yaml.load(reader, Loader=yaml.BaseLoader)

    valid_keys = ['rename', 'drop_coords', 'units']
    for key in metadata_dict.keys():
        if not key in valid_keys:
            raise KeyError(f'Invalid metadata key: {key}')

    if 'rename' in metadata_dict:
        for orig_var, target_var in metadata_dict['rename'].items():
            try:
                ds = ds.rename({orig_var: target_var})
            except ValueError:
                pass

    if 'drop_coords' in metadata_dict:
        for drop_coord in metadata_dict['drop_coords']:
            if drop_coord in ds.coords:
                ds = ds.drop(drop_coord)

    if 'units' in metadata_dict:
        for var, units in metadata_dict['units'].items():
            ds[var].attrs['units'] = units

    return ds


def get_new_log(infile_logs=None, repo_dir=None):
    """Generate command log for output file.

    Args:
      infile_logs (dict) : keys are file names,
        values are the command log
    """

    try:
        repo = git.Repo(repo_dir)
        repo_url = repo.remotes[0].url.split('.git')[0]
    except git.exc.InvalidGitRepositoryError:
        repo_url = None
    new_log = cmdprov.new_log(code_url=repo_url,
                              infile_logs=infile_logs)

    return new_log


def zip_zarr(zarr_filename, zip_filename):
    """Zip a zarr collection"""
    
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as fh:
        for root, _, filenames in os.walk(zarr_filename):
            for each_filename in filenames:
                each_filename = os.path.join(root, each_filename)
                fh.write(each_filename, os.path.relpath(each_filename, zarr_filename))


def to_zarr(ds, filename):
    """Write to zarr file"""
                
    for var in ds.variables:
        ds[var].encoding = {}

    if filename[-4:] == '.zip':
        zarr_filename = filename[:-4]
    else:
        zarr_filename = filename

    ds.to_zarr(zarr_filename, mode='w', consolidated=True)
    if filename[-4:] == '.zip':
        zip_zarr(zarr_filename, filename)
        shutil.rmtree(zarr_filename)
