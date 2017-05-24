import json
from os.path import dirname, exists, join

import numpy as np
import pandas as pd
import xarray as xr


BASE_URL = 'http://data.stats.gov.cn/english/easyquery.htm'
CACHE_DIR = join(dirname(__file__), 'cn_nbs')


def load_nbs(level, series, period, cache_dir=CACHE_DIR):
    cache_path = join(cache_dir, '%s_%s_%s.json' % (level, series, period))
    if exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        return parse_nbs_json(raw)
    else:
        return load_nbs_web(level, series, period, cache_dir)

# curl 'http://data.stats.gov.cn/english/easyquery.htm?m=QueryData&dbcode=hgnd&rowcode=zb&colcode=sj&wds=%5B%5D&dfwds=%5B%7B%22wdcode%22%3A%22sj%22%2C%22valuecode%22%3A%221995-2014%22%7D%5D&k1=1472763426686' -H 'Accept: application/json, text/javascript, */*; q=0.01' -H 'Referer: http://data.stats.gov.cn/english/easyquery.htm?cn=C01' -H 'X-Requested-With: XMLHttpRequest' -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2837.0 Safari/537.36' --compressed

# curl 'http://data.stats.gov.cn/english/easyquery.htm?m=QueryData&dbcode=hgnd&rowcode=zb&colcode=sj&wds=%5B%5D&dfwds=%5B%7B%22wdcode%22%3A%22zb%22%2C%22valuecode%22%3A%22A090302%22%7D%5D&k1=1472763563287' -H 'Cookie: JSESSIONID=DC6EE94A71D8597AB0B249B7337EFB79; experience=show; u=6' -H 'DNT: 1' -H 'Accept-Encoding: gzip, deflate, sdch' -H 'Accept-Language: en-GB,en;q=0.8,en-US;q=0.6,en-CA;q=0.4' -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2837.0 Safari/537.36' -H 'Accept: application/json, text/javascript, */*; q=0.01' -H 'Referer: http://data.stats.gov.cn/english/easyquery.htm?cn=C01' -H 'X-Requested-With: XMLHttpRequest' -H 'Connection: keep-alive' --compressed


def load_nbs_web(level, series, periods, cache_dir=None):
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
    # Example from http://data.stats.gov.cn/english/easyquery.htm?m=QueryData&dbcode=fsnd&rowcode=reg&colcode=sj&wds=[{"wdcode":"zb","valuecode":"A090201"}]&dfwds=[{"wdcode":"sj","valuecode":"1995-2014"}]&k1=1472740901192

    # curl 'http://data.stats.gov.cn/english/easyquery.htm?m=QueryData&dbcode=fsnd&rowcode=reg&colcode=sj&wds=%5B%7B%22wdcode%22%3A%22zb%22%2C%22valuecode%22%3A%22A090201%22%7D%5D&dfwds=%5B%7B%22wdcode%22%3A%22sj%22%2C%22valuecode%22%3A%222013-2014%22%7D%5D&k1=1472747318027' -H 'Accept: application/json, text/javascript, */*; q=0.01' -H 'Referer: http://data.stats.gov.cn/english/easyquery.htm?cn=E0103' -H 'X-Requested-With: XMLHttpRequest' -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2837.0 Safari/537.36' --compressed
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
    ds = xr.Dataset()

    returncode = data['returncode']
    assert returncode == 200

    # Metadata
    wdnodes = data['returndata']['wdnodes']
    for dimension in wdnodes:
        wdcode = dimension['wdcode']
        codes = []
        for node in dimension['nodes']:
            codes.append(node['code'])
        ds[wdcode] = codes
        ds[wdcode].attrs['wdname'] = dimension['wdname']
        ds.set_coords(wdcode)

    shape = [c.size for _, c in ds.coords.items()]
    ds['data'] = (ds.coords, np.ones(shape) * np.nan)
    ds['hasdata'] = (ds.coords, np.zeros(shape, dtype=bool))

    datanodes = data['returndata']['datanodes']
    for datanode in datanodes:
        wds = {}
        for wd in datanode['wds']:
            wds[wd['wdcode']] = wd['valuecode']
        dim = tuple([wds[wdcode] for wdcode in ds.coords])
        if datanode['data']['hasdata']:
            ds['hasdata'].loc[dim] = True
            ds['data'].loc[dim] = datanode['data']['data']
        else:
            pass

    rename = {'zb': 'indicator', 'sj': 'period', 'reg': 'region'}
    ds = ds.rename({k: v for k, v in rename.items() if k in ds.coords})

    freq = {
        'Year': 'A',
        }[ds['period'].attrs['wdname']]
    periods = pd.to_datetime(ds['period']).to_period(freq)
    ds['period'] = periods
    ds = ds.reindex(period=sorted(periods))

    if 'region' in ds.coords:
        ds['region'] = [int(r) for r in ds['region'].values]

    return ds
