__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import json
import subprocess
import time


def probe_stream(filename):
    """
    Returns a blank dictionary if no stream available
    Quicker than OpenCV ->takes >30 seconds to time out
    ffprobe uses on ffprobe.exe on local drive
    """
    # make sure to download ffprobe.exe
    cmnd = [r'C:\ffmpeg-2022\bin\ffprobe.exe', '-show_format', '-pretty', '-loglevel', 'quiet', '-of', 'json', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s = p.communicate()[0]

    if p.returncode == 0:

        out, err = p.communicate()

        p.wait()
        probe_dct = json.loads(out)

        try:
            if probe_dct:
                return True
            elif err:
                print(err)
        except Exception as e:
            print(e)
    else:
        print(p.returncode)
        time.sleep(5)