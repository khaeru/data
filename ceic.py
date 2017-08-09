#!/usr/bin/env python3
"""CEIC Data China Premium Database"""
import logging
from os.path import commonprefix, exists, dirname, join
import pickle

import pandas as pd  # >= 0.20
import pint
import xarray as xr
# Importing data also requires: gb2260, tqdm

__all__ = [
    'import_ceic',
    'load_ceic',
    ]

CACHE_FORMAT = 'pkl'  # Either 'pkl' or 'nc'

DATA_DIR = join(dirname(__file__), 'ceic')

VERBOSE = False

# Used by import_ceic and lookup_gbcode
META = {}

log = logging.getLogger(__name__)


class VariableImportError(Exception):
    """Something went wrong importing a variable."""
    pass


def import_ceic(filename=None, input_dir=DATA_DIR, output_dir=DATA_DIR,
                cache=True):
    """Import CEIC Data.

    Returns an xarray.Dataset containing data imported from
    *input_dir*/*filename*.tsv, in TSV format, containing the metadata columns:

    - name
    - country
    - frequency
    - unit
    - source
    - status
    - series
    - code
    - function
    - info
    - first obs
    - last obs
    - updated

    …plus any number of data columns with names parseable as dates.

    The data import is controlled by *input_dir*/*filename*.yaml, in YAML
    format, containing:

    - name_map: A mapping from city names in the input data, to official names
      found in the gb2260 database.
    - variables: A mapping from short names to tuples matching part or all of
      the input data in the 'name' column.

    If *debug* is True (default: False), verbose debugging information is
    logged to the 'ceic' logger and also written to *output_dir*/ceic.log.
    """
    from glob import glob

    from tqdm import tqdm

    # Read options from file
    _read_metadata(input_dir)

    # List of pd.DataFrames
    dfs = []

    # Read data from each CSV file in input_dir
    log.info('Read data')
    for fn in glob(join(input_dir, '*.csv')):
        log.info('  from %s', fn)

        # Read the CSV fill, drop empty rows and columns, and remove exact
        # duplicate rows
        tmp = pd.read_csv(fn).dropna(axis=[0, 1], how='all') \
                .drop_duplicates() \
                .rename(columns={'': 'Name', 'Unnamed: 0': 'Name'})

        log.info('  %d series', len(tmp))

        dfs.append(tmp)

    # Concatenate to a single pd.DataFrame
    data = pd.concat(dfs)

    # NB to debug, subset the data here like:
    #     l = ['MWD', 'NUN']
    #     data = data[data['Series Code'].str.match('.*CHA(%s)' % '|'.join(l))]
    # or:
    #     data = data[:5000]

    # Count non-zero values
    data_cols = pd.to_datetime(data.columns, errors='coerce').notnull()
    N = data.loc[:, data_cols].notnull().sum().sum()
    log.info('Total %d series, %d non-zero values', len(data), N)

    # Rename columns to lower case
    data.columns = data.columns.to_series().apply(str.lower)

    log.info('Transform metadata columns')

    # Strip a preceding 'CN: ' from the name column, convert to list
    data['name'] = data['name'].str.lstrip('CN:').str.lstrip().str.split(': ')

    # Convert the series code column ('1234 (ABCD)') and split into separate
    # columns
    sc_regex = '(?P<series_code_numeric>\d+) \((?P<series_code_alpha>\w+)\)'
    sc_rename = {
        'series_code_numeric': 'series code numeric',
        'series_code_alpha': 'series code alpha',
        }
    data = pd.concat([
        data,
        data['series code'].str.extract(sc_regex, expand=True),
        ], axis=1).rename(columns=sc_rename)

    # Convert types
    for s in ('first', 'last'):
        col = '%s obs. date' % s
        data[col] = pd.to_datetime(data[col], errors='coerce')

    # Make the alpha series code the index
    data.set_index('series code alpha', drop=False)

    log.info('Preprocess rules from file:')
    for rule in META['preprocess']:
        log.info(rule)

        # Transform the rule into Python expressions
        predicate = eval('lambda row: %s' % rule['predicate'])
        env = {}
        exec('def transform(row):\n %s\n return row' % rule['transform'], env)

        # Apply the rule
        mask = data.apply(predicate, axis=1)
        data[mask] = data[mask].apply(env['transform'], axis=1)

    # Add columns 'name group', 'region', 'gbcode', 'level'
    log.info('Compute indexes')

    if not VERBOSE:
        META['progressbar'] = tqdm(total=len(data))
    data = pd.concat([data, data.apply(_match_gbcode, axis=1)], axis=1)

    if not VERBOSE:
        META['progressbar'].close()

    # commented: for debugging only, see below
    # log.debug('Name part searches 1: %r', _search_kwargs_debug1)
    # log.debug('Name part searches 2: %r', _search_kwargs_debug2)

    log.info('Group into variables')
    # regional detail → list of xr.DataArray
    das = {level: [] for level in (0, 1, 2, 3)}
    groups = data.groupby('name group')
    if not VERBOSE:
        groups = tqdm(groups)
    for key, df in groups:
        # Accumulate messages to log on error
        message = [
            'Variable: %s' % ': '.join(key),
            '%s series' % len(df),
            ]

        try:
            # Handle duplicate series
            df = _deduplicate(df)

            # Index on GB codes
            df = df.set_index('gbcode', drop=False)

            # A mask for data columns plus 'level'
            data_cols = pd.to_datetime(df.columns, errors='coerce').notnull()

            try:
                df = _normalize_units(df, data_cols)
            except pint.errors.UndefinedUnitError as e:
                raise VariableImportError('Skip (unknown units): %s' % e)

            # Construct metadata
            attrs, name = _make_attrs(df, key)
        except VariableImportError as e:
            log.warning('\n  '.join(message + list(e.args)))
            continue

        # Select values (discard metadata except 'level'), and drop nulls
        data_cols = data_cols | (df.columns == 'level')
        values = df.iloc[:, data_cols].dropna(axis=1, how='all')

        # Convert column index to pd.Period
        values.columns = values.columns.astype(pd.Period)
        values.columns.name = 'period'
        values.index.name = 'gbcode'

        # Group to national, provincial, etc. levels
        for level, level_df in values.groupby('level'):
            # Convert to an xarray object
            if level < 1:
                level_df = level_df.iloc[0, :]
            das[level].append(xr.DataArray(level_df, attrs=attrs, name=name))

    # Done grouping

    # For each level of aggregation, merge the xr.DataArrays into one
    # xr.Dataset and cache it
    dss = {}
    count = dict(variables=0, values=0)

    for level, arrays in das.items():
        log.info('At administrative level %d', level)

        if len(arrays) == 0:
            log.info('No variables.')
            continue

        ds = xr.merge(arrays).dropna(dim='period', how='all')

        log.info('Number of variables: %d', len(ds))
        # log.debug('Resulting xarray.Dataset:\n%s', ds)  # *extremely* verbose

        # Count variables and non-null values
        count['variables'] += len(ds)
        notnull = ds.notnull().sum()
        values = sum([notnull[x] for x in notnull])
        count['values'] += values

        log.info('Number of non-null values: %d', values)
        # log.debug(notnull)  # *extremely* verbose

        if not cache:
            log.info('Skip output to cache')
            continue

        _serialize_units(ds)

        out_fn = join(output_dir, 'ceic-{}.{}'.format(level, CACHE_FORMAT))
        log.info('Write to %s', out_fn)

        if CACHE_FORMAT == 'nc':
            ds.to_netcdf(out_fn)
        elif CACHE_FORMAT == 'pkl':
            pickle.dump(ds, open(out_fn, 'wb'))

        log.info('…done.')
        dss[level] = ds

    log.info('Imported %d variables, %d values (%.2f%% of input)',
             count['variables'], count['values'], 100 * count['values'] / N)

    # Return the dataframe
    return dss


def load_ceic(input_dir=DATA_DIR):
    """Load CEIC Data.

    Data is returned as an xarray.Dataset.

    If cached data exists in *input_dir*/ceic-[0123].nc, it is read and
    returned. Otherwise, the data is imported by calling import_ceic(), with
    *debug* as an argument.
    """
    # Names of cached files
    filenames = [join(input_dir, 'ceic-{}.{}'.format(level, CACHE_FORMAT))
                 for level in (0, 1, 2, 3)]

    if not all(map(exists, filenames)):
        import_ceic(input_dir=input_dir, output_dir=input_dir)

    dss = {}  # Dictionary of xr.Dataset

    for level, filename in enumerate(filenames):
        log.debug('Read from %s…', filename)

        if CACHE_FORMAT == 'nc':
            ds = xr.open_dataset(filename)
        elif CACHE_FORMAT == 'pkl':
            ds = pickle.load(open(filename, 'rb'))

        _unserialize_units(ds, level == 0)

        log.debug('…done.')

        dss[level] = ds

    return dss


def _deduplicate(df):
    """Handle duplicates in *df*, returning a deduplicated pd.DataFrame."""
    skip = False

    duplicates = {}
    for dupe_key, dupe_df in df.groupby('gbcode'):
        if len(dupe_df) == 1:
            continue

        duplicates[dupe_key] = len(dupe_df)

    if skip:
        raise VariableImportError('Skip (duplicates): %s' % ' '.join(
            ['%d (%d)' % (k, v) for k, v in duplicates.items()]))

    df = df.drop_duplicates('gbcode')  # FIXME
    if len(df) == 0:
        raise VariableImportError('Skip (0 rows after dropping duplicates')

    return df


def _get_name(name):
    i = META['prefixes'][name]
    META['prefixes'][name] += 1
    return '%s_%d' % (name, i) if i else name


def _load_units(defs):
    """Set up a pint.UnitRegsitry from the string unit *defs*."""
    from io import StringIO

    import pint

    global META

    # Load units
    META['ureg'] = pint.UnitRegistry()
    META['ureg'].load_definitions(StringIO(defs))


def _make_attrs(df, key):
    """Construct attributes for a xr.DataArray from *df* and *key*."""
    attrs = {
        'name': ': '.join(key),
        'source': ', '.join(df['source'].unique()),
        'first obs': str(df['first obs. date'].min()),
        'last obs': str(df['last obs. date'].max()),
        'updated': df['last update time'].max(),
        'series codes': ' '.join(df['series code alpha']),
        }

    attrs['unit'] = df['unit'].iloc[0].to_base_units()

    # It's an error to have non-unique values for any of these
    for a in ['country', 'frequency', 'status']:
        unique_value = df[a].str.lower().unique()
        if len(unique_value) == 1:
            attrs[a] = unique_value[0]
        else:
            raise VariableImportError('Skip (multiple values for %s): %s'
                                      % (a, unique_value))

    # Get the alias for this variable
    name = META['rename variables'].get(key, None)

    if name is None:
        # No alias.
        codes = list(df['series code alpha'])

        # Use _get_name to construct a name
        name = _get_name(commonprefix(codes))

        if len(codes) > 1:
            log.debug('Use common prefix %s for codes: %s', name,
                      ' '.join(codes))

    return attrs, name


# For disambiguating
_search_kwargs = [
    ('name_pinyin', dict(partial=True)),
    ('name_pinyin', dict(partial=True, level='highest')),  # e.g. Beijing
    ('name_en', dict(partial=True, level='highest')),  # e.g. Inner Mongolia
    ]
_match_cache = {}

# For 26481 series                   Without caching:      With caching:
# _search_kwargs_debug1 = [0, 0, 0]  # [77551 52170 30765] [11536 11474 6406]
# _search_kwargs_debug2 = [0, 0, 0]  # [21405 15342  3909] [   18    16    0]


def _match_gbcode(row):
    """Identify the Chinese administrative region for *row*.

    Returns a pd.Series with the keys:
    - gbcode
    - level
    - name group
    - region

    """
    import gb2260

    def _recurse(name, root=False):
        """Recursive method for matching *name*."""
        part = META['rename regions'].get(name[-1], name[-1])
        info = None
        gbcode = 0
        level = 0

        # First step: try to look up the last part of *name*
        ambiguous = False
        for key, kwargs in _search_kwargs:
            # NB for debugging, loop on enumerate(_search_kwargs) and update
            # _search_kwargs_debug1
            kwargs[key] = part
            try:
                div = gb2260.divisions.search(**kwargs)
            except gb2260.AmbiguousRegionError:
                # At least one match for *part*, but impossible to know
                # which one
                ambiguous = True
                continue
            except:
                # Not ambiguous
                continue
            else:
                gbcode, level = div.code, div.level
                break

        if not gbcode and not ambiguous:
            # Not a region name → the remaining parts of *name* are the
            # variable group. Return here.
            return {
                'name group': tuple(name),
                'gbcode': 0,
                'level': 0,
                'region': [],
                }

        # Either *part* was matched, or it was ambiguous. Recursively match
        # the next part(s) of *name*
        info = _recurse(name[:-1])

        # info now contains all information about the leftwards parts of
        # *name*, including any parent regions. Append this region's name
        info['region'].append(part)

        if ambiguous:
            # encountered AmbiguousRegionError on first try, above. Try to
            # disambiguate now
            for key, kwargs in _search_kwargs:
                # NB for debugging, loop on enumerate(_search_kwargs) and
                # update _search_kwargs_debug2
                kwargs[key] = part
                if info['gbcode'] > 0:
                    # Parent information was returned with a gbcode; that must
                    # be the parent of *part*
                    kwargs['within'] = info['gbcode']

                try:
                    div = gb2260.divisions.search(**kwargs)
                except:
                    continue
                else:
                    gbcode, level = div.code, div.level
                    break

            # Serious problem if we haven't found anything by this point
            if not gbcode:
                msg = ("Couldn't match '%s' in %s under parent %d" %
                       (part, row['name'], info['gbcode']))
                raise ValueError(msg)

        # Overwrite the gbcode and level from the recursion (if any) with the
        # values determined at this level
        info['gbcode'] = gbcode
        info['level'] = level

        # Back up to the previous level
        return info

    # Try the cache first
    result = None
    for key in map(lambda i: row['name'][-i:], (1, 2, 3)):
            try:
                result = _match_cache[':'.join(key)]
            except KeyError:
                continue
            else:
                result = dict(zip(('gbcode', 'level', 'region'), result))
                result['name group'] = tuple(row['name'][:-len(key)])
                break

    if not result:
        # Kick off the recursion
        result = _recurse(row['name'], root=True)

        # Convert region to tuple
        result['region'] = tuple(result['region'])

        # Cache the result
        if result['gbcode'] > 0:
            cache_val = (result['gbcode'], result['level'], result['region'])
            _match_cache[':'.join(result['region'])] = cache_val

    # Rename according to metadata
    result['rename'] = META['rename variables'].get(
        result['name group'], row['series code alpha'])

    if not VERBOSE:
        META['progressbar'].update(1)

    return pd.Series(result)


def _normalize_units(df, data_cols):
    """Return a version of *df* with the same units in all rows."""
    df.loc[:, 'unit'] = df['unit'].apply(_parse_unit)
    common = df['unit'].min()

    def _check_and_norm(u):
        factor = (u / common).to_base_units()
        assert factor.unitless
        return factor.magnitude

    factor = df['unit'].apply(_check_and_norm)

    if (factor.unique() == [1.]).all():
        return df
    else:
        log.debug('  Normalizing units for %s series to %s',
                  (factor != 1).sum(), common)
        log.debug('  By factors: %s', factor.unique())
        df.iloc[:, data_cols] = df.iloc[:, data_cols].mul(factor, axis=0)

        # Can't simply assign here; see
        # https://github.com/hgrecco/pint/issues/401
        for label in df.index:
            df.set_value(label, 'unit', common)
        return df


def _parse_unit(text):
    """Convert *text* into a pint.Quanitity."""
    text = text.lower().replace(' ', '_').replace('%', 'percent')
    return META['ureg'](text)


def _read_metadata(input_dir):
    """Read metadata.yaml"""
    from collections import defaultdict
    import yaml

    global META

    options_fn = join(input_dir, 'metadata.yaml')
    log.info('Read metadata from %s', options_fn)

    with open(options_fn) as f:
        META.update(yaml.load(f))

    META['prefixes'] = defaultdict(int)

    # Load units
    _load_units(META['units'])

    # Invert this mapping
    META['rename variables'] = {tuple(v): k for k, v in
                                META['rename variables'].items()}


def _serialize_units(ds):
    for _, var in ds.data_vars.items():
        var.attrs['unit'] = str(var.attrs['unit'])
    ds.attrs['units'] = META['units']


def _unserialize_units(ds, update_registry=False):
    if update_registry:
        _load_units(ds.attrs['units'])

    for _, var in ds.data_vars.items():
        var.attrs['unit'] = META['ureg'](var.attrs['unit'])


if __name__ == '__main__':
    import click

    try:
        from _util import click_nowrap
        click_nowrap()  # Prettier help output from click
    except ImportError:  # User hasn't downloaded _util.py
        pass

    @click.group()
    @click.option('--verbose', is_flag=True, help='Give verbose output')
    def cli(verbose):
        global VERBOSE

        VERBOSE = verbose

        # Configure logging
        logging.getLogger().setLevel(logging.NOTSET)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        log.addHandler(handler)

    @cli.command(name='import')
    @click.option('--no-cache', 'no_cache', help="don't cache imported data")
    def _import(no_cache, output_dir=DATA_DIR):
        """Import data from CSV and update the cache."""
        from datetime import datetime

        # Also log to file
        log_fn = join(output_dir, datetime.now().isoformat() + '.log')
        handler = logging.FileHandler(log_fn, mode='w')
        log.addHandler(handler)

        import_ceic(output_dir=output_dir, cache=not no_cache)

    @cli.command()
    def load():
        """Load data from the cache and exit."""
        dss = load_ceic()
        print(dss.keys())
        print(dss[3])

    cli()
