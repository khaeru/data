from . import requirements

if __name__ == '__main__':
    for source, sreq in requirements.items():
        print('\n'.join(map(lambda s: s.replace(' ', ''), sreq)))
