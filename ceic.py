"""CEIC Data China Premium Database"""
from collections import OrderedDict
import logging
import os

import gb2260
import pandas as pd

__all__ = [
    'import_ceic',
    'load_ceic',
    ]

DATA_DIR = os.path.join(os.path.dirname(__file__), 'ceic')
CURRENT = '2014-03-14'

# Used by import_ceic and lookup_gbcode
NAME_MAP = {}
units = None

log = logging.getLogger(__name__)


def import_ceic(filename=CURRENT, input_dir=DATA_DIR, output_dir=DATA_DIR,
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
    from ast import literal_eval

    import xarray as xr
    import yaml

    global NAME_MAP

    # Configure logging
    handler = logging.FileHandler(os.path.join(output_dir, filename + '.log'),
                                  mode='w')
    log.addHandler(handler)

    # Variables to import
    v = {}

    # Read options
    options_fn = os.path.join(input_dir, filename + '.yaml')
    log.info('reading options from %s', options_fn)
    with open(options_fn) as f:
        options = yaml.load(f)
        NAME_MAP.update(options['name_map'])
        load_units(options['units'])
        v.update(options['variables'])

    # Sort variables *in reverse* by the string representation of the name
    # fragment. This ensures that e.g. ['Population', 'Rural'] is matched
    # before ['Population'].
    variables = OrderedDict(sorted(v.items(),
                                   key=lambda t: ''.join(t[1]),
                                   reverse=True))

    # Converters for use in reading the TSV. Interpret strings like:
    #   ('No of Motor Vehicle', 'Total')
    # as Python tuples.
    converters = {
        'name': lambda x: literal_eval(x),
        'series code': lambda x: literal_eval(x)[1],
        }

    data_fn = os.path.join(input_dir, filename + '.tsv')
    log.info('read data from %s', data_fn)

    # Read the TSV file, indexing on the CEIC series code
    # - Drop empty rows and columns
    # - Remove exact duplicate rows
    data = pd.read_csv(data_fn, delimiter='\t', index_col='series code',
                       converters=converters) \
             .dropna(axis=[0, 1], how='all').drop_duplicates()

    N_series = data.shape[0]
    log.info('  %d series', N_series)

    # The 'name' column contains values like:
    #     ('No of Motor Vehicle', 'Private Owned', 'Hebei', 'Baoding')
    #
    # *variables* contains a list of name fragments, like:
    #     ('No of Motor Vehicle', 'Private Owned')
    #
    # If the name fragment matches, keep the series; otherwise, discard. Use
    # the remaining elements of the name to identify the region for the data.
    log.info('match %d variables', len(variables))

    # Dict of variable name → list of (gbcode, series.name)
    index = {k: [] for k in variables.keys()}

    def match_series(series):
        """Return True if *series* is to be imported.

        match_series() also populates *index*[*var*].
        """
        name = series['name']

        # Loop over variables
        for var, fragment in variables.items():
            # Find the position of *fragment* in *name*
            i = tuple_find(fragment, name)

            if i < 0:
                # No match found
                continue

            # Identify region
            gb = lookup_gbcode(list(name[i+len(fragment):]))
            if gb is not None:
                # Region identified; record the CEIC series code (note this is
                # NOT the same as series['name']!) and keep the series.
                index[var].append((gb, series.name))
                return True
            else:
                log.debug('Could not match GB code for %s',
                          name[i+len(fragment):])

        # Either no variable matched, or could not identify region; discard
        return False

    # Compute a boolean vector on the data & use it to subset all series
    data = data[data.apply(match_series, axis=1)]
    log.info('Dropped %d series; %d remain', N_series - data.shape[0],
             data.shape[0])
    data['unit'] = data['unit'].apply(parse_unit)

    # Split the data to several variables, populating *das*
    # index[k][1] contains a list of column names for each variable; transfer
    # these from *data* into a number of separate data frames in *d*.
    das = []  # list of xarray.DataArray
    for k, v in index.items():  # Iterate over variables
        log.info('Processing %d series for %s', len(v), k)

        if len(v) == 0:
            continue

        new, old = zip(*v)  # new ← GB/T 2260 codes; old ← CEIC codes

        # Select the series for this variable
        # - Drop any empty rows or columns
        # - Rename rows with GB/T 2260 codes
        df = data.loc[old, :].dropna(axis=[0, 1], how='all') \
                 .rename(index=dict(zip(old, new)))

        dropped = len(new) - df.shape[0]
        if dropped > 0:
            log.info('Dropped %s empty series', dropped)

        # Resolve duplicate series names. Some series have the same CEIC name
        # and region, but different CEIC codes, data, and metadata. This
        # results in multiple columns named with the same GB/T 2260 codes.
        dupes = df.index.get_duplicates()

        if len(dupes):
            log.info('Deduplicating %d duplicate series names', len(dupes))
        for dupe in dupes:
            # Select each set of duplicates
            dseries = df.loc[dupe, :].copy()

            # Count empty elements in each
            dseries['na_count'] = dseries.notnull().sum(axis=1)

            # log.debug('Merging duplicates:\n%s', dseries.T)

            # Sort from most to least full, then merge values
            dseries = dseries.sort_values(by='na_count', ascending=False) \
                             .drop('na_count', axis=1) \
                             .fillna(method='ffill', axis=0)
            dseries.iloc[0, :] = dseries.iloc[-1, :]

            # Merge metadata
            dseries['source'] = ', '.join(dseries['source'])
            dseries['first obs'] = dseries['first obs'].min()
            dseries['last obs'] = dseries['last obs'].max()
            dseries['updated'] = dseries['updated'].max()

            # Overwrite both the original copies with new data
            df.loc[dupe, :] = dseries

        # Drop the duplicate rows, keepeing only one of each
        df = df[~df.index.duplicated()]

        log.info('%d series remaining', df.shape[0])

        i = df.columns.get_loc('updated') + 1

        # if k == 'pop':
        #     log.info('  adjust population to common units → 10³')
        #     prov = d['pop']['unit'] == 'Person mn'
        #     d['pop'].ix[prov, i:] *= 1e3
        #     d['pop'].loc[prov, 'unit'] = 'Person th'

        log.info('  convert to xarray.DataArray')

        df = normalize_units(df)

        # Construct metadata
        attrs = {}

        try:
            attrs['unit'] = df['unit'].iloc[0].to_base_units()
        except AttributeError:
            print(df['unit'].iloc[0], k)
            attrs['unit'] = ''

        for a in ['country', 'frequency', 'status']:
            unique_value = df[a].unique()
            assert len(unique_value) == 1, 'multiple values for %s: %s' % (
                a, unique_value)
            attrs[a] = unique_value[0]
        attrs['source'] = ', '.join(df['source'].unique())
        attrs['first obs'] = df['first obs'][df['first obs'] > '0000'].min()
        attrs['last obs'] = df['last obs'].max()
        attrs['updated'] = df['updated'].max()
        attrs['name'] = ', '.join(variables[k])

        # Split to data and metadata
        values = df.iloc[:, i:]
        values.index = values.index.astype(pd.Period)
        values.index.name = 'gbcode'
        values.columns.name = 'period'

        # Convert to an xarray object
        das.append(xr.DataArray(values, attrs=attrs, name=k))

    # Merge the xr.DataArrays into a single xr.Dataset
    ds = xr.merge(das)
    log.debug(' resulting xarray.Dataset:\n%s', ds)
    log.info(' number of non-null values:\n%s', ds.notnull().sum())

    # Cache output
    if cache:
        out_fn = os.path.join(output_dir, 'ceic.nc')
        log.info('write to %s', out_fn)

        # Serialize units
        for _, var in ds.data_vars.items():
            var.attrs['unit'] = str(var.attrs['unit'])
        ds.attrs['units'] = options['units']

        ds.to_netcdf(out_fn)
    else:
        log.info('skip output to cache')

    log.info('done.')

    return ds


def load_ceic(input_dir=DATA_DIR):
    """Load CEIC Data.

    Data is returned as an xarray.Dataset.

    If cached data is stored 'ceic.nc' in the directory *input_dir*, this
    data is read and returned. Otherwise, the data is imported by calling
    import_ceic(), with *debug* as an argument.
    """
    import xarray as xr

    cache_fn = os.path.join(input_dir, 'ceic.nc')
    log.debug('Reading from %s…', cache_fn)
    try:
        result = xr.open_dataset(cache_fn)
        log.debug('…done')
        log.debug('Unserializing units')
        load_units(result.attrs['units'])
        for _, var in result.data_vars.items():
            var.attrs['unit'] = units(var.attrs['unit'])
    except RuntimeError:
        log.debug('Missing, importing directly')
        result = import_ceic()
    return result


def normalize_units(df):
    i = df.columns.get_loc('updated') + 1
    common = df['unit'].min()

    def _check_and_norm(u):
        factor = (u / common).to_base_units()
        assert factor.unitless
        return factor.magnitude

    factor = df['unit'].apply(_check_and_norm)

    if (factor.unique() == [1.]).all():
        return df
    else:
        log.info('  normalizing units for %s series to %s',
                 (factor != 1).sum(), common)
        log.info('  by factors: %s', factor.unique())
        df.iloc[:, i:] = df.iloc[:, i:].mul(factor, axis=0)

        # Can't simply assign here; see
        # https://github.com/hgrecco/pint/issues/401
        for label in df.index:
            df.set_value(label, 'unit', common)
        return df


def parse_unit(text):
    return units(text.lower().replace(' ', '_'))


def load_units(defs):
    from io import StringIO
    import pint

    global units

    units = pint.UnitRegistry()
    units.load_definitions(StringIO(defs))


def lookup_gbcode(name):
    """Return a GB/T 2260 code for tuple *name*.

    If *name* is a 2-tuple, the entries are assumed to be (province, city),
    indicating which province the city is in, and used to constrain the
    lookup. If it is a 1-tuple, the GB/T 2260 code for the single entry is
    returned. Names are translated using *name_map* before lookup.

    Returns None for all other or invalid values.

    """
    result = None
    name_en = None
    try:
        # If only one division name is given, assume it is a province
        # TODO make this more flexible for e.g. airport series
        kwargs = dict(level=1)
        if len(name) > 1:
            kwargs['within'] = gb2260.lookup(name_en=name[-2], level=1)
            kwargs['level'] = 2
        name_en = NAME_MAP.get(name[-1], name[-1])
        result = gb2260.lookup(name_en=name_en, **kwargs)
    except LookupError:
        pass

    # commented: extremely verbose
    # if result is not None:
    #     log.debug("found code %d for %s" % (result, name))

    return result


def tuple_find(a, b):
    """Find tuple in tuple.

    If tuple *a* appears, in order, within tuple *b*, the index of a[0] in
    b; else -1.

    """
    a = tuple(a)
    b = tuple(b)
    l_a = len(a)
    for i in range(len(b) - l_a + 1):
        if b[i:i+l_a] == a:
            return i
    return -1


if __name__ == '__main__':
    import click

    @click.group()
    @click.option('--verbose', is_flag=True, help='Give verbose output')
    def cli(verbose):
        # Configure logging
        log.setLevel(logging.DEBUG if verbose else logging.INFO)

    @cli.command(name='import')
    @click.option('--no-cache', 'no_cache', help="don't cache imported data")
    def _import(no_cache):
        """Import data from TSV and update the cache."""
        import_ceic(cache=not no_cache)

    @cli.command()
    def load():
        """Load data from the cache and exit."""
        ds = load_ceic()
        print(ds, ds['pop'], sep='\n')

    cli()
