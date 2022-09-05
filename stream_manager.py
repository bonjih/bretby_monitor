__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import json
import subprocess


def probe_stream(filename):
    """
    Returns a blank dictionay if no stream available
    Quicker than OpenCV, takes 20 seconds to time out
    ffprobe uses on ffprobe.exe
    """
    # make sure to download ffprobe.exe
    cmnd = [r'C:\ffmpeg-2022\bin\ffprobe.exe', '-show_format', '-pretty', '-loglevel', 'quiet', '-of', 'json', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    p.wait()
    probe_dct = json.loads(out)

    if probe_dct:
        return True
    elif err:
        print(err)
