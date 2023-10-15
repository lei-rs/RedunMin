import os


def is_master():
    try:
        return os.environ['RANk'] == '0'
    except KeyError:
        return True
