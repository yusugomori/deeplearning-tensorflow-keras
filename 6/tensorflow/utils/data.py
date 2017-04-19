import os
from urllib.request import urlretrieve

base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_file(filename, url=None, datadir=None):
    if url is None:
        raise
    if datadir is None:
        datadir = base_dir
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    fpath = os.path.join(datadir, filename)

    download = False
    if os.path.exists(fpath):
        pass
    else:
        download = True

    if download:
        print('Downloading data from', url)
        try:
            try:
                urlretrieve(url, fpath)
            except URLError as e:
                raise
            except HTTPError as e:
                raise
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    return fpath
