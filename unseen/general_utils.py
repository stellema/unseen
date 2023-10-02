"""General utility functions."""

import argparse
import re

import numpy as np
import xclim
from scipy.stats import genextreme as gev


class store_dict(argparse.Action):
    """An argparse action for parsing a command line argument as a dictionary.

    Examples
    --------
    precip=mm/day becomes {'precip': 'mm/day'}
    ensemble=1:5 becomes {'ensemble': slice(1, 5)}
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, val = value.split("=")
            if ":" in val:
                start, end = val.split(":")
                try:
                    start = int(start)
                    end = int(end)
                except ValueError:
                    pass
                val = slice(start, end)
            else:
                try:
                    val = int(val)
                except ValueError:
                    pass
            getattr(namespace, self.dest)[key] = val


def convert_units(da, target_units):
    """Convert units.

    Parameters
    ----------
    da : xarray DataArray
        Input array containing a units attribute
    target_units : str
        Units to convert to

    Returns
    -------
    da : xarray DataArray
       Array with converted units
    """

    xclim_unit_check = {
        "deg_k": "degK",
        "kg/m2/s": "kg m-2 s-1",
    }

    if da.attrs["units"] in xclim_unit_check:
        da.attrs["units"] = xclim_unit_check[da.units]

    try:
        da = xclim.units.convert_units_to(da, target_units)
    except Exception as e:
        in_precip_kg = da.attrs["units"] == "kg m-2 s-1"
        out_precip_mm = target_units in ["mm d-1", "mm day-1"]
        if in_precip_kg and out_precip_mm:
            da = da * 86400
            da.attrs["units"] = target_units
        else:
            raise e

    return da


def date_pair_to_time_slice(date_list):
    """Convert two dates to a time slice object.

    Parameters
    ----------
    date_list : list or tuple
        Start and end date in YYYY-MM-DD format

    Returns
    -------
    time_slice : slice
        Slice from start to end date
    """

    assert len(date_list) == 2
    start_date, end_date = date_list

    date_pattern = "([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})"
    assert re.search(date_pattern, start_date), "Start date not in YYYY-MM-DD format"
    assert re.search(date_pattern, end_date), "End date not in YYYY-MM-DD format"

    time_slice = slice(start_date, end_date)

    return time_slice


def event_in_context(data, threshold, direction):
    """Put an event in context.

    Parameters
    ----------
    data : numpy ndarray
        Population data
    threshold : float
        Event threshold
    direction : {'above', 'below'}
        Provide statistics for above or below threshold

    Returns
    -------
    n_events : int
        Number of events in population
    n_population : int
        Size of population
    return_period : float
        Return period for event
    percentile : float
        Event percentile relative to population (%)
    """

    n_population = len(data)
    if direction == "below":
        n_events = np.sum(data < threshold)
    elif direction == "above":
        n_events = np.sum(data > threshold)
    else:
        raise ValueError("""direction must be 'below' or 'above'""")
    percentile = (np.sum(data < threshold) / n_population) * 100
    return_period = n_population / n_events

    return n_events, n_population, return_period, percentile


def fit_gev(data, user_estimates=[], generate_estimates=False):
    """Fit a GEV by providing fit and scale estimates.

    Parameters
    ----------
    data : numpy ndarray
    user_estimates : list, optional
        Estimate of the location and scale parameters
    generate_estimates : bool, default False
        Fit GEV to data subset first to estimate parameters (useful for large datasets)

    Returns
    -------
    shape : float
        Shape parameter
    loc : float
        Location parameter
    scale : float
        Scale parameter
    """

    if user_estimates:
        loc_estimate, scale_estimate = user_estimates
        shape, loc, scale = gev.fit(data, loc=loc_estimate, scale=scale_estimate)
    elif generate_estimates:
        shape_estimate, loc_estimate, scale_estimate = gev.fit(data[::2])
        shape, loc, scale = gev.fit(data, loc=loc_estimate, scale=scale_estimate)
    else:
        shape, loc, scale = gev.fit(data)

    return shape, loc, scale


def return_period(data, event):
    """Get return period for given event by fitting a GEV"""
    
    shape, loc, scale = fit_gev(data, generate_estimates=True)
    probability = gev.sf(event, shape, loc=loc, scale=scale)
    return_period = 1. / probability
    
    return return_period


def gev_return_curve(data, event_value, bootstrap_method='non-parametric', n_bootstraps=1000):
    """Return x and y data for a GEV return period curve.

    Parameters
    ----------
    data : xarray DataArray
    event_value : float
        Magnitude of event of interest
    bootstrap_method : {'parametric', 'non-parametric'}, default 'non-parametric'
    n_bootstraps : int, default 1000 

    """

    # GEV fit to data
    shape, loc, scale = fit_gev(data, generate_estimates=True)
    
    curve_return_periods = np.logspace(0, 4, num=10000)
    curve_probabilities = 1.0 / curve_return_periods
    curve_values = gev.isf(curve_probabilities, shape, loc, scale)
    
    event_probability = gev.sf(event_value, shape, loc=loc, scale=scale)
    event_return_period = 1. / event_probability
    
    # Bootstrapping for confidence interval
    boot_values = curve_values
    boot_event_return_periods = []
    for i in range(n_bootstraps):
        if bootstrap_method == 'parametric':
            boot_data = gev.rvs(shape, loc=loc, scale=scale, size=len(data))
        elif bootstrap_method == 'non-parametric':
            boot_data = np.random.choice(data, size=data.shape, replace=True)
        boot_shape, boot_loc, boot_scale = fit_gev(boot_data, generate_estimates=True)

        boot_value = gev.isf(curve_probabilities, boot_shape, boot_loc, boot_scale)
        boot_values = np.vstack((boot_values, boot_value))
        
        boot_event_probability = gev.sf(event_value, boot_shape, loc=boot_loc, scale=boot_scale)
        boot_event_return_period = 1. / boot_event_probability
        boot_event_return_periods.append(boot_event_return_period)

    curve_values_lower_ci = np.quantile(boot_values, 0.025, axis=0)
    curve_values_upper_ci = np.quantile(boot_values, 0.975, axis=0)
    curve_data = curve_return_periods, curve_values, curve_values_lower_ci, curve_values_upper_ci
    
    boot_event_return_periods = np.array(boot_event_return_periods)
    boot_event_return_periods = boot_event_return_periods[np.isfinite(boot_event_return_periods)]
    event_return_period_lower_ci = np.quantile(boot_event_return_periods, 0.025)
    event_return_period_upper_ci = np.quantile(boot_event_return_periods, 0.975)
    event_data = event_return_period, event_return_period_lower_ci, event_return_period_upper_ci
    
    return curve_data, event_data


def plot_gev_return_curve(
    ax, data, event_value, bootstrap_method='parametric', n_bootstraps=1000, ylabel=None
):
    """Plot a single return period curve.

    Parameters
    ----------
    data : xarray DataArray
    """

    curve_data, event_data = gev_return_curve(
        data,
        event_value,
        bootstrap_method=bootstrap_method,
        n_bootstraps=n_bootstraps,
    )
    curve_return_periods, curve_values, curve_values_lower_ci, curve_values_upper_ci = curve_data
    event_return_period, event_return_period_lower_ci, event_return_period_upper_ci = event_data
    
    ax.plot(
        curve_return_periods,
        curve_values,
        color='tab:blue',
        label='GEV fit to data'
    )
    ax.fill_between(
        curve_return_periods,
        curve_values_lower_ci,
        curve_values_upper_ci,
        color='tab:blue',
        alpha=0.2,
        label='95% CI on GEV fit'
    )
    ax.plot(
        [event_return_period_lower_ci, event_return_period_upper_ci],
        [event_value] * 2,
        color='0.5',
        marker='|',
        linestyle=':',
        label='95% CI for record event',
    )
    print(f'{event_return_period:.0f} year return period')
    print(f'95% CI: {event_return_period_lower_ci:.0f}-{event_return_period_upper_ci:.0f} years')
    empirical_return_values = np.sort(data, axis=None)[::-1]
    empirical_return_periods = len(data) / np.arange(1.0, len(data) + 1.0)
    ax.scatter(
        empirical_return_periods,
        empirical_return_values,
        color='tab:blue',
        alpha=0.5,
        label='empirical data'        
    )
    
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[3], handles[0], handles[1], handles[2]]
    labels = [labels[3], labels[0], labels[1], labels[2]]
    ax.legend(handles, labels, loc='upper left')
    ax.set_xscale("log")
    ax.set_xlabel("return period (years)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ylim = ax.get_ylim()
    ax.set_ylim([50, ylim[-1]])
    ax.grid()


