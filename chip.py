#!/usr/bin/env python3
"""Import, merge & preprocess raw data from the China Household Income Project

http://www.ciidbnu.org/chip/index.asp?lang=EN

TECHNICAL INFORMATION

CHIP data sets are organized by three dimensions, each with a string label:

1. *sample*: e.g. 'rural', 'urban' or 'migrant'
2. *unit* of measurement/analysis: e.g. 'person' or 'household'.
3. *section* of a questionnaire or survey: e.g. 'abc' or 'income and assets'.

Some CHIP surveys contain multiple sections which apply to individual people,
but the units of observation are distinct: for instance, all household members,
children of the household head(s) who are not resident in the household, and
parents of the household head(s) who may or may not be resident in the
household. These are usually indexed by different variables and have different
data associated with them, so chip.py stores them separately.

FILES IN chip/

- rar/ — original files downloaded from the CHIP data website
- cache/ — CHIP data imported to Python/pandas format
- YYYY/ — unpacked files for year YYYY in Stata .dta format (data) and .pdf
  (questionnaires and summary statistics)
- YYYY.yaml — metadata for the year YYYY data set
- YYYY.log — log files for import
"""
import logging
import os
import re

import pandas as pd
import pytest

__all__ = [
    'import_chip',
    'load_chip',
    ]

# Path containing raw CHIP data and data set description (.yaml) files
RAW = os.path.join(os.path.dirname(__file__), 'chip')

# Target path for imported data
CACHE = os.path.join(RAW, 'cache')

log = logging.getLogger(__name__)


def _deduplicate(x, log, log_indent='\t'):
    """Drop duplicate rows in *x*, logging the result."""
    shape = x.shape
    x.drop_duplicates(inplace=True)
    dropped = shape[0] - x.shape[0]
    if dropped > 0:
        log.info('%sDropped %d duplicate rows' % (log_indent, dropped))


def _get_dataset_info(filename):
    """Retrieve data set information from *filename*."""
    from yaml import load

    info = load(open(filename))

    # Directory containing the data set files
    path = info['files']['path']
    path = path if isinstance(path, list) else [path]
    path = os.path.join(*map(str, path))

    # Compile regular expressions for data file names
    fn_re = info['files']['name']
    fn_re = fn_re if isinstance(fn_re, list) else [fn_re]
    fn_re = map(re.compile, fn_re)

    # Mapping information used to determine the dimensions of data in a file
    fn_map = info.get('map', {})

    # Information on columns
    column_info = info.get('column', {})
    # …type conversions
    convert = {k: v for k, v in column_info.items() if 'type' in v}
    # …columns to drop in the input
    drop = [k for k, v in column_info.items() if v.get('drop', False)]

    # Information about units of observation
    units = info.get('unit', {})

    return path, fn_re, fn_map, convert, drop, units


def _map_dimensions(initial, dim_map):
    """Return (unit, sample, section) for any *initial* values."""
    if dim_map is None:
        return initial

    dims = ['unit', 'sample', 'section']
    mapped = {d: v for d, v in initial.items() if
              (d in dims and v is not None)}

    dim = dim_map.get('_dim')
    default = dim_map.get('_default', {})

    # Is the key multidimensional?
    if isinstance(dim, list):
        # Yes: look up a submap on the first dimension
        submap = dim_map[initial[dim[0]]]
        # Get the submap's values
        values = submap.get(initial.get(dim[1]), {})
        values.update(submap.get('_all', {}))
        # Get the submap's defaults
        default.update(submap.get('_default', {}))
    elif dim is None:
        values = None
    else:
        # Look up using a single dimension
        values = dim_map[initial[dim]]

    # If the mapped values are a dict(), update the result
    if isinstance(values, dict):
        mapped.update(values)
    else:
        mapped[dim] = values

    # Add defaults
    result = tuple([mapped.get(d, default.get(d)) for d in dims])

    assert not any(map(lambda x: x is None, result))
    return result


def _read_stata_drop(filename, drop):
    """Read *filename* in Stata format, optionally *drop*-ing some columns."""
    # Determine a list of columns to read. Default None → all columns
    columns = None

    # Dropping involves opening the file twice: once to create the list of
    # columns, once to actually read the data. Only do so if there are
    # columns to be dropped.
    if len(drop) > 0:
        # Read the list of columns in a safe mode
        itr = pd.read_stata(filename, convert_categoricals=False, chunksize=1)
        # The first chunk (1 row) gives the column names
        all_cols = next(itr).columns
        # Close the file
        del itr

        # List of columns, excluding any dropped columns
        columns = [c for c in all_cols if c not in drop]

    return pd.read_stata(filename, columns=columns)


def _report_duplicates(data, index, log):
    """*log* information about duplicates in *index* columns of *data*."""
    from textwrap import indent

    # Boolean mask of rows that have the same values in the columns *index*
    dupe_index = data.duplicated(subset=index, keep=False)

    # All data for duplicate rows
    dupes = data[dupe_index].groupby(index)

    log.info('\tIndex contains %d duplicates.' % len(dupes))
    log.debug('\tShowing only mismatched variables:')

    # Iterate over sets of duplicate lines
    for key, rows in dupes:
        key = key if isinstance(key, tuple) else [key]
        keys = ['%s=%s' % (k, v) for k, v in zip(index, key)]
        log.debug('\n\t' + ', '.join(keys))

        # Rows where duplicates differ. NB assumes only two rows in each set
        # of duplicates
        different = rows.iloc[0, :] != rows.iloc[1, :]
        with pd.option_context('display.width', 1000):
            log.debug(indent(str(rows.transpose()[different]
                                     .dropna(how='all')), '\t'))


def import_chip(year, input_dir='.', output_dir=None, only=None, debug=False):
    """Import the CHIP data set for *year* to files *output_dir*/*year*….pkl.

    In addition to writing a .pkl file for every (unit, sample) combination in
    the data set, returns a tuple containing:

    1. a dictionary of pandas.DataFrames in which every key is a CHIP (unit,
       sample), and data from CHIP sections are joined together on matching
       indices.
    2. a list of files from which the data were imported.

    *input_dir*/*year*.yaml, in YAML format, is read for information on the
    data set. The information file MUST contain the `file` key, with two
    subkeys:

    - `path`: list of subdirectories below *input_dir* containing the CHIP .dta
      files for *year*. For example, to specify `*input_dir*/foo/bar`:

      file:
        path:
          - foo
          - bar

    - `filename`: list of regular expressions for matching data file names. The
      regexes MAY contain named groups for each dimension, e.g.:

      …(?P<sample>…)…(?P<unit>…)…(?P<section>…).dta

      The parts of the filename captured by these groups are used for the
      dimension names, after being mapped (below).

    The information file MAY contain the following keys that control how data
    from the files is imported:

    - `map`: mapping describing how to convert (unit, sample, section) names
      matched by the filename expressions to other values. MUST contain '_dim'
      naming 1 or 2 of the three dimensions; this is the dimension used to look
      up matched names in the map. Example:

      map:
        _dim: [sample, section]
        _default: {unit: household}
        R:
          _all: {sample: rural}
          _default: {unit: other}
          d: {unit: r_child}
          e1: {unit: r_nonres_child}

      For the example above:

      - All files matching sample='R' are assigned sample='rural'.
      - Files matching sample='R' and section='d' are assigned unit='r_child'.
      - Files matching sample='R' but not section in ['d', 'e1'] are assigned
        unit='other'.
      - Files not matching sample='R' are assigned unit='household'.

      If parsing of the map does not yield a name for each dimension (unit,
      sample, section ) for any file, a KeyError is raised.

    - `column`: maps column names to information about columns in input data
      files. For each column name, if the information contains the key 'type',
      then the column is converted to that type on import. If it contains the
      key 'drop', the column is not imported. Example:

      column:
        a1:
          name: Verbose description of this column, optional
          type: int
        b2:
          drop: true

    - `unit`: maps unit names to information about indexes. For each unit name,
      if the information contains the key 'index', the single column or list of
      columns named are used as the index for all data from that unit of
      observation. If the information also contains 'unique: true' (default),
      the index must be unique; non-unique index data raises an exception. If
      'unique: false', non-unique index data does not raise an exception, but
      information on duplicate entries is logged. Example:

      unit:
        hh_head_parent:
          index: [hhcode, e3]
        household:
          index: hhcode
          unique: false

    Other entries in the information file are ignored.

    *only* is an optional collection of (unit, sample) tuples. If given, only
    those specified combinations from the data set are imported.

    Information is logged to *output_dir*/*year*.log. If *debug* is True
    (default: False), verbose debugging information is logged.

    Example:
        chip.import_chip(2002, debug=True)
    """
    import logging

    # Default options
    if output_dir is None:
        output_dir = input_dir
    elif not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Configure logging
    logging.basicConfig(format='%(message)s',
                        level=logging.DEBUG if debug else logging.INFO)
    log = logging.getLogger('import_chip')
    log_handler = logging.FileHandler(os.path.join(output_dir,
                                                   '%d.log' % year),
                                      mode='w')
    log.addHandler(log_handler)

    # Data set information file
    info = _get_dataset_info(os.path.join(input_dir, '%d.yaml' % year))
    path, fn_re, fn_map, convert, drop, units = info

    # Apply the file name expression to all files in the path, filtering
    # out those that do not match
    all_files = sorted(os.listdir(os.path.join(input_dir, path)))
    files = []
    for pattern in fn_re:
        files.extend(filter(None, map(pattern.match, all_files)))

    log.info('Matched %d files for CHIP %d' % (len(files), year))
    log.info('\t' + '\n\t'.join(map(lambda x: x.string, files)) + '\n')

    # Temporary containers for loaded data
    data = {}

    # Iterate over data files
    for match in files:
        # Extract dimension labels from the filename, map and defaults
        unit, sample, section = _map_dimensions(match.groupdict(), fn_map)

        # If omitting some portions of the data set, skip this data file
        if only is not None and (unit, sample) not in only:
            log.info('Skipping (%s, %s, %s)\n  from %s\n  not in %s' %
                     (unit, sample, section, match.string, only))
            continue
        else:
            log.info('Loading (%s, %s, %s) from %s' %
                     (unit, sample, section, match.string))

        # Read the .dta file and drop indicated columns
        raw = _read_stata_drop(os.path.join(input_dir, path, match.string),
                               drop)
        raw.name = sample

        log.info('\t{} rows, {} columns'.format(*raw.shape))
        log.debug('\tColumns: %s' % ' '.join(raw.columns))

        # Drop duplicate rows
        _deduplicate(raw, log)

        # Convert column data types
        for col, col_info in convert.items():
            if col in raw:
                raw[col] = raw[col].astype(col_info['type'])

        # Store with other data of the same (unit, sample)
        if (unit, sample) not in data:
            # This is the first/only section of data for this (unit, sample)
            data[(unit, sample)] = raw
        else:
            # Join with previously-imported data for this (unit, sample)
            d = data[(unit, sample)]
            common = d.columns.intersection(raw.columns)

            log.info(('\tMerging with existing data on: %s'
                      '\t%d rows before') % (' '.join(common), len(d)))

            data[(unit, sample)] = d.merge(raw, how='outer')

            log.info('\t%d rows after' % len(data[(unit, sample)]))

        log.info('')

    # Process the merged data
    for key in sorted(data.keys()):
        log.info('Finalizing %s' % str(key))

        # Set the index of the data. NB this is not done through the 'index'
        # argument to pandas.read_stata, so that the column type can be changed
        # by the above code
        unit_info = units.get(key[0], {})
        index = unit_info.get('index')
        if index is not None:
            index = index if isinstance(index, list) else [index]

            # Should table indices be unique?
            unique = unit_info.get('unique', True)

            log.info('\t%snique index on: %s' %
                     ('U' if unique else 'Non-u', ' '.join(index)))

            try:
                # If unique == True, this will raise a ValueError
                new = data[key].set_index(index, verify_integrity=unique)
                if not unique and not new.index.is_unique:
                    # Report duplicates even if unique == False
                    raise ValueError
            except ValueError:
                _report_duplicates(data[key], index, log)

                # If the data set information states this index should be
                # unique, this is a fatal error
                if unique:
                    raise
            finally:
                data[key] = new

        # Write to file
        fn = os.path.join(output_dir, '%d-%s-%s.pkl' % (year, *key))
        log.info('\tWriting to %s\n' % fn)
        data[key].to_pickle(fn)

    log.removeHandler(log_handler)

    return data.keys(), files


def load_chip(year, input_dir=CACHE, import_dir=RAW, only=None, debug=False):
    """Load the CHIP data set for *year*.

    Returns a dictionary of pandas.DataFrames in which every key is a CHIP
    (unit, sample).

    If the data has already been imported and cached in *input_dir* (using
    `import_chip()`), the .pkl files in this directory are loaded. Otherwise,
    `import_chip()` is called to import raw CHIP .dta files from *import_dir*
    to *input_dir*. If *debug* is True (default: False), verbose debugging
    information is logged during import.

    *only* is an optional collection of (unit, sample) tuples. If given, only
    those specified combinations from the data set are returned.

    Example:
        d = chip.load_chip(2002, only=[('person', 'migrant'),
                                       ('household', 'urban')])
    """
    # Directory and filenames
    fn_re = re.compile('^%d-(?P<unit>.*)-(?P<sample>.*)\.pkl$' % year)

    def find_files():
        try:
            result = filter(None, map(fn_re.match, os.listdir(input_dir)))
        except FileNotFoundError:
            os.mkdir(input_dir)
            result = []
        return list(result)

    # Check for existing data files
    files = find_files()

    if len(files) == 0:
        # Data has not been imported from raw .dta
        import_chip(year, input_dir=import_dir, output_dir=input_dir,
                    only=only, debug=debug)

    # Data has been imported from raw .dta to .pkl, read those directly
    data = {}
    for f in find_files():
        key = (f.group('unit'), f.group('sample'))
        if only is not None and key not in only:
            continue
        fn = os.path.join(input_dir, f.string)
        data[key] = pd.read_pickle(fn)

    return data


# Tests for the above code
TEST_YEARS = {
    1995: {
        'files': 4,  # number of .dta files in the data set
        'data': 4,   # number of unique combinations of (unit, sample) in the
                     # data set
        },
    1999: {'data': 2, 'files': 2},
    2002: {'data': 7, 'files': 10},
    2007: {'data': 20, 'files': 30},
    2008: {'data': 11, 'files': 26},
    2013: {'data': 6, 'files': 21},
    }


class TestCHIP:
    """Test suite for import_chip and load_chip."""
    # Temporary directory for importing and loading data
    DIR = 'chip_test'

    @pytest.mark.parametrize('year', TEST_YEARS.keys())
    def test_import(self, year):
        data, files = import_chip(year, input_dir=RAW, output_dir=self.DIR,
                                  debug=True)
        # Check the number of .dta files read
        assert len(files) == TEST_YEARS[year]['files']
        # Check the number of unique combinations of (unit, sample)
        assert len(data) == TEST_YEARS[year]['data']

    @pytest.mark.parametrize('year', TEST_YEARS.keys())
    def test_load(self, year):
        data = load_chip(year, input_dir=self.DIR, debug=True)
        assert len(data) == TEST_YEARS[year]['data']

    def test_import_only_2013(self):
        data, files = import_chip(2013, input_dir=RAW, output_dir=self.DIR,
                                  only=[('household', 'urban')], debug=True)
        assert len(data) == 1
        assert len(files) == 21

    def test_load_only_2013(self):
        data = load_chip(2013, input_dir=self.DIR,
                         only=[('household', 'urban')], debug=True)
        assert len(data) == 1

    @classmethod
    def teardown_class(cls):
        """Remove the temporary directory."""
        from shutil import rmtree
        rmtree(cls.DIR)


if __name__ == '__main__':
    import click

    try:
        from _util import click_nowrap
        click_nowrap()
    except ImportError:  # User hasn't downloaded _util.py
        pass

    @click.group(help=__doc__)
    @click.option('--verbose', is_flag=True, help='Give verbose output')
    def cli(verbose):
        # Configure logging
        log.setLevel(logging.DEBUG if verbose else logging.INFO)

    @cli.command(name='import', help=import_chip.__doc__)
    @click.argument('years', nargs=-1, required=True)
    def _import(years):
        if years == ['all']:
            # Import all years
            years = TEST_YEARS.keys()
        else:
            years = list(map(int, years))

        log.info('Importing %s', years)
        for year in years:
            import_chip(year, input_dir=RAW, output_dir=CACHE, debug=True)

    cli()
