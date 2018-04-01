#!/usr/bin/env python3
"""National Bureau of Statistics of China."""
import json
from os.path import dirname, exists, join

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    'load_nbs',
    ]

BASE_URL = 'http://data.stats.gov.cn/english/easyquery.htm'
CACHE_DIR = join(dirname(__file__), 'cn_nbs')


def load_nbs(level, series, period, cache_dir=CACHE_DIR, offline=False):
    cache_path = join(cache_dir, '%s_%s_%s.json' % (level, series, period))
    if exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        return parse_nbs_json(raw)
    elif not offline:
        return load_nbs_web(level, series, period, cache_dir)


def load_nbs_web(level, series, periods, cache_dir=CACHE_DIR):
    """Load the China National Bureau of Statistics web data API.

    *level* must be either 'national' or 'regional'. *series* must be a string
    indicating the data series requested, e.g. 'A090302'. *periods* must be a
    list of periods with one or more entries, in the forms indicated by the web
    interface:
    - 1995
    - 2003,2012
    - LATEST10
    - LAST5

    Returns an xarray.Dataset containing the requested data. If *cache_dir* is
    a directory (default: None), raw data returned by the JSON API is saved in
    a file named '*cache_dir*/*level*_*series*_*periods*.json'.

    load_nbs_web() does not currently support:
    - quarterly or monthly data, or
    - data at aggregations other than national or regional.

    """
    # Example query string (decoded):
    # http://data.stats.gov.cn/english/easyquery.htm?m=QueryData
    #   &dbcode=fsnd
    #   &rowcode=reg
    #   &colcode=sj
    #   &wds=[{"wdcode":"zb","valuecode":"A090201"}]
    #   &dfwds=[{"wdcode":"sj","valuecode":"1995-2014"}]
    #   &k1=1472740901192
    from datetime import datetime
    from requests import Request, Session

    # Parameters for constructing the query string
    params = {
        # Method of easyquery.htm to call
        'm': 'QueryData',
        # Periods are always one dimension of the returned data
        'colcode': 'sj',
        # Timestamp
        'k1': int(datetime.now().timestamp() * 1000),
        }

    # Wrap series and periods in the form expected by the query string
    _series = {'wdcode': 'zb', 'valuecode': series}
    _periods = {'wdcode': 'sj', 'valuecode': periods}

    wds = []
    dfwds = []

    # Set the database code, second dimension (rows in the web display), and
    # data selectors
    if level == 'regional':
        # Regional data:
        params['dbcode'] = 'fsnd'
        params['rowcode'] = 'reg'
        # Page selector or data series drop-down
        wds = [_series]
        # Column dimension of data
        dfwds = [_periods]
    elif level == 'national':
        params['dbcode'] = 'hgnd'
        params['rowcode'] = 'zb'
        # Two dimensional data: leave this blank
        wds = []
        # Select both series and periods
        dfwds = [_series, _periods]
    else:
        raise ValueError('level must be one of: national, regional.')

    # Convert the wds and dfwds parameters to stringified JSON
    seps = (',', ':')
    params['wds'] = json.dumps(wds, separators=seps)
    params['dfwds'] = json.dumps(dfwds, separators=seps)

    # Prepare the HTTP request
    prepped = Request('GET', BASE_URL, params=params).prepare()

    # Print the complete query string for debugging
    print(prepped.url)

    # Retrieve the data
    result = Session().send(prepped)

    # Cache data if requested
    if cache_dir is not None:
        cache_fn = join(cache_dir, '%s_%s_%s.json' % (level, series, periods))
        with open(cache_fn, 'w') as f:
            json.dump(result.json(), f, indent=2)

    # Parse the returned data
    return parse_nbs_json(result.json())


def parse_nbs_json(data):
    """Parse *data* for a single indicator."""
    assert data['returncode'] == 200, 'Data was produced by a failed request'

    ds = xr.Dataset()
    da_attrs = {}

    # Read dimension information
    for dim_info in data['returndata']['wdnodes']:
        # This is one of:
        # reg (region) = region
        # sj (shíjiān, time period) = period
        # zb (zhǐbiāo, index) = indicator
        dim_id = dim_info['wdcode']

        # List of codes along this dimension
        codes = []
        for node in dim_info['nodes']:
            code = node['code']
            codes.append(code)

            # For indicators, also store metadata
            if dim_id == 'zb':
                da_attrs[code] = {}
                for key, value in node.items():
                    # Skip empty metadata
                    if value != '':
                        da_attrs[code][key] = value

        # Make the dimension a coordinate in the xr.Dataset
        ds[dim_id] = codes
        ds.set_coords(dim_id)

        # Name of this dimension
        ds[dim_id].attrs['wdname'] = dim_info['wdname']

    # Compute the shape of the data
    coords = []
    shape = []
    for name, coord in ds.coords.items():
        if name != 'zb':
            coords.append(name)
            shape.append(coord.size)

    # Allocate one xr.DataArray for each indicator
    for zb in ds['zb'].values:
        ds[zb] = (coords, np.ones(shape) * np.nan)
        # Save attributes
        ds[zb].attrs = da_attrs[zb]

    # Drop the indicator dimension
    ds = ds.drop('zb')

    # Iterate over data points
    for obs in data['returndata']['datanodes']:
        # Skip observations with no data
        if not obs['data']['hasdata']:
            continue
        else:
            value = obs['data']['data']

        # Assemble the coordinates of this data point
        wds = {wd['wdcode']: wd['valuecode'] for wd in obs['wds']}

        # Separate the indicator
        zb = wds.pop('zb')

        # Retrieve the dimensions matching the coords
        dim = tuple([wds[wdcode] for wdcode in ds.coords])

        # Store
        ds[zb].loc[dim] = value

    # Rename the dimensions using the more descriptive 'wdname'
    rename = {}
    for dim_name, da in ds.coords.items():
        rename[dim_name] = da.attrs['wdname'].lower()
    # …except use common "period" instead of "year", "month", etc.
    rename['sj'] = 'period'
    ds = ds.rename(rename)

    # Convert frequency and sort by date
    freq = {
        'Year': 'A',
        }[ds['period'].attrs['wdname']]
    periods = pd.to_datetime(ds['period']).to_period(freq)
    ds['period'] = periods
    ds = ds.reindex(period=sorted(periods))

    # Convert region codes to integers
    if 'region' in ds.coords:
        ds['region'] = [int(r) for r in ds['region'].values]

    return ds


if __name__ == '__main__':
    import click

    try:
        from _util import click_nowrap
        click_nowrap()
    except ImportError:  # User hasn't downloaded _util.py
        pass

    cli = click.Group('cli', help=__doc__)

    def common_options(f):
        f = click.argument('series', nargs=1)(f)
        f = click.option('--period', 'period', required=True)(f)
        f = click.option('--level', 'level', required=True,
                         type=click.Choice(['national', 'regional']))(f)
        return f

    @cli.command()
    @common_options
    def fetch(level, period, series):
        result = load_nbs_web(level, series, period)
        print(result)

    @cli.command()
    @common_options
    def dump(level, period, series):
        result = load_nbs(level, series, period, offline=True)
        for name, da in result.data_vars.items():
            print(name, da)

    cli()
