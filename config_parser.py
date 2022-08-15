__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

#############################################
# load config.json
# if required, change variables in config.json
#############################################
import json

configs = {}


def config_parser():
    with open('configs.json', 'r') as jsonFile:
        data = json.load(jsonFile)
        configs.update(data.items())
    return configs


def db_parser():
    db_field_key = []

    with open('db_fields.json', 'r') as jsonFile:
        data = json.load(jsonFile)
        for key, value in data.items():
            db_field_key.append(key)
    return db_field_key
