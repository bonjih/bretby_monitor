__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

from config_parser import config_parser

config = config_parser()


class ConfigDict(dict):
    """
    creates a dictionary from jconfig.json
    """

    # GELPhotos = config['image_path']  # image from GelPhotos folder
    # file_type = config['file_type']
    # filtered = config['filtered']  # where scripts puts new images to process cv
    # user = config['user']
    # passwd = config['passwd']
    # host = config['host']
    # database = config['database']
    # known_distance = config['known_distance']
    # known_width = config['known_width']

    def __init__(self):
        super().__init__()
        self.config = dict()

    def add(self, key, value):
        self[key] = value


dict_obj = ConfigDict()


def get_values():
    for key, value in config.items():
        dict_obj.add(key, value)
    return list(dict_obj.values())


def get_keys():
    for key, value in config.items():
        dict_obj.add(key, value)
    return list(dict_obj.keys())

