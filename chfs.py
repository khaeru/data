"""China Household Finance Survey

This code expects the six data download files offered by the CHFS website to be
in the 'chfs/rar' subdirectory.
1. On the command line, run the commands 'extract' then 'import_chfs' to
   prepare the data.
2. (Optionally) Use the commands 'check_cache' and 'check_encoding'.
3. Import this file ("import chfs") in a Python script and use the method
   load_chfs() to retrieve data in a pandas DataFrame.
"""
from itertools import product
from os.path import abspath, dirname, join

import click
import pandas as pd

DATA_DIR = abspath(join(dirname(__file__), 'chfs'))

__all__ = [
    'check_cache',
    'check_encoding',
    'extract',
    'import_chfs',
    ]


@click.group(help=__doc__)
def cli():
    pass


@cli.command()
def check_cache():
    """Load the cached datafiles and perform some sanity checks."""
    for lang, unit in product(('eng', 'chn'),
                              ('hh', 'ind', 'master')):
        data = {fmt: _read_cache(unit, lang, fmt) for fmt in ('csv', 'dta')}

        if unit == 'master':
            data['sas'] = _read_cache(unit, lang, 'sas')

        print('\n', unit, lang)

        try:
            # print((data['csv'] == data['dta']).all(axis=0).T)
            print(data['csv'].head(), data['dta'].head(), sep='\n')
        except Exception as e:
            print(data['csv'].columns, data['dta'].columns,
                  # data['csv'].index, data['dta'].index,
                  )
            continue

        if unit == 'master':
            # print((data['sas'] == data['csv']).all(axis=0).T)
            # print((data['sas'] == data['dta']).all(axis=0).T)
            print(data['sas'].head())


@cli.command()
@click.option('--lang', default='eng', help='language of CSV file to check')
def check_encoding(lang):
    """Check the encoding of the CSV source files.

    The CHFS CSV source data is in an unknown encoding. chardet (
    http://chardet.readthedocs.io) gives windows-1252 with low confidence, and
    *any* of the Python standard encodings (
    https://docs.python.org/3/library/codecs.html#standard-encodings) fails on
    at least some fields in the data files.

    Use this command to prepare lists of lines to exclude, for metadata.yaml,
    e.g.:

    $ python3 -m chfs check_encoding | cut -f1 | tr "\n" ","

    """
    def _decode(b):
        """Try to decode bytes *b* under multiple encodings."""
        for _encoding in ['gb18030']:
            # 'utf-8', 'gb2312', 'hz', 'big5',
            # 'big5hkscs', 'cp950', 'iso2022_jp_2'
            try:
                return b.decode(_encoding), _encoding
            except UnicodeDecodeError as e:
                error = e
        raise error

    # (line, field #) → offending bytes
    errors = {}

    # List of field names, to identify field #
    field_names = []

    DELIM = bytes('\t', 'utf-8') if lang == 'eng' else bytes(',', 'utf-8')

    # Read the file and split into lines
    f = open(join(DATA_DIR, 'hh_release_%s_20130109.csv' % lang), 'rb')
    lines = f.read().split(b'\x0D\x0A')

    # Iterate over lines
    for l, line in enumerate(lines):
        try:
            # Try to decode this line. If all encodings fail, an exception is
            # thrown
            text, _ = _decode(line)

            # First line of the file contains field names
            if l == 0:
                field_names = text.split('\t' if lang == 'eng' else ',')
        except UnicodeDecodeError as e1:
            # Determine the field(s) on this line containing offending bytes
            for f, field in enumerate(line.split(DELIM)):
                try:
                    _decode(field)
                except UnicodeDecodeError as e2:
                    # Store information about the offending field
                    errors[(l, f+1, field_names[f])] = (field, e2)

    # Process errors
    for k, v in sorted(errors.items()):
        # Identify the offending character
        e = v[1]
        offending = e.object[e.start:e.end]

        # # Description of where the offending byte is found:
        # initial = e.start == 1
        # terminal = e.end == len(v[0]) - 1
        # position = 'initial' if initial and not terminal else ''
        # position = 'terminal' if not initial and terminal else position
        # position = 'internal' if not (initial and terminal) else position

        # Output some diagnostic information
        print('\t'.join(map(str, k)),
              v[0],
              offending,
              # position,
              sep='\t')

        # # Try to 'repair' the field content by deleting the offending byte.
        # # Since the byte might be part of an offending *sequence*, this is
        # # prohibitively complicated: https://en.wikipedia.org/wiki/GB_18030
        # try:
        #     print('  deletion works:',
        #           _decode(e.object.replace(offending, b'')))
        # except UnicodeDecodeError:
        #     print('  deletion FAILS!')
        #     pass


@cli.command()
def extract():
    """Extract and combine the archives into a single directory."""
    import os
    from os.path import splitext
    import subprocess

    # Get list of files
    for fn in os.listdir(join(DATA_DIR, 'rar')):
        if not splitext(fn)[1] == '.rar':
            continue
        # Extract. Can't use rarfile for this, yet:
        # https://github.com/markokr/rarfile/issues/36
        subprocess.call(['unar', '-f', '-o', DATA_DIR, join(DATA_DIR, fn)])

    for dn, subdirs, fns in os.walk(DATA_DIR):
        if dn == DATA_DIR:
            to_delete = subdirs
            continue
        for fn in fns:
            os.replace(join(dn, fn), join(DATA_DIR, fn))

    for dn in to_delete:
        os.rmdir(join(DATA_DIR, dn))


def _read_cache(unit, lang, fmt):
    """Read a file from the cache."""
    import pickle

    cache_fn = join(DATA_DIR, '-'.join(('cache', unit, lang, fmt)) + '.pkl')

    with open(cache_fn, 'rb') as f:
        return pickle.load(f)


def _read_file(unit='hh', lang='eng', fmt='csv', **kwargs):
    """Read in one of the CHFS data files, returning a pandas.DataFrame.

    - *units*: units of observation, 'hh', 'ind' (individual), or 'master'.
    - *lang*: language, 'eng', or 'chn'.
    - *format*: 'csv', 'dta' (Stata), or 'sas'.
    """
    # Construct the filename
    parts = [unit, 'release']
    if unit != 'master':
        parts.append(lang)
    parts.append('20130109')
    fn = '_'.join(parts) + '.' + ('sas7bdat' if fmt == 'sas' else fmt)

    data_fn = join(DATA_DIR, fn)

    # Choose or define a method to read the file
    if fmt == 'dta':
        method = pd.read_stata
    elif fmt == 'sas':
        method = pd.read_sas
    elif fmt == 'csv':
        if unit == 'master' or lang == 'chn':
            sep = ','
        else:
            sep = '\t'

        def method(fn, **_kwargs):
            # Chardet suggests the encoding is windows-1252
            return pd.read_csv(fn, sep=sep, encoding='gb18030',
                               low_memory=False, **_kwargs)

    # Read the file
    print(kwargs)
    result = method(data_fn, **kwargs)

    # SAS only: convert column names to lower case
    if fmt == 'sas':
        result.columns = map(str.lower, result.columns)

    # Set an index on the data
    result['hhid'] = result['hhid'].astype(int)

    if unit == 'hh':
        result.set_index('hhid', inplace=True, verify_integrity=True)
    else:
        result['pline'] = result['pline'].astype(int)
        result.set_index(['hhid', 'pline'], inplace=True,
                         verify_integrity=True)

    # Sort the data
    result.sort_index(inplace=True)

    return result


@cli.command()
def import_chfs():
    """Import data from the raw CHFS files to the cache."""
    import pickle

    import yaml

    options = yaml.load(open(join(DATA_DIR, 'metadata.yaml')))

    for unit, lang, fmt in product(('hh', 'ind', 'master'),
                                   ('eng', 'chn'),
                                   ('csv', 'dta', 'sas',)):
        key = '-'.join((unit, lang, fmt))
        try:
            # Read the data
            print('\nLoading', key)
            data = _read_file(unit, lang, fmt, **options[key])
            print('  %d rows' % len(data))

            # Write to cache
            with open(join(DATA_DIR, 'cache-%s.pkl' % key), 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    cli()
