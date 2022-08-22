import json
import subprocess


def probe_stream(filename):
    """
    Returns a black dictionay if no stream available
    Quicker than OpenCV, takes 20 seconds to time out
    ffprobe uses on ffprobe.exe
    """
    cmnd = [r'C:\ffmpeg-2022\bin\ffprobe.exe', '-show_format', '-pretty', '-loglevel', 'quiet', '-of', 'json', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out, err = p.communicate()
    p.wait()
    probe_dct = json.loads(out)

    if probe_dct:
        return True
    elif err:
        print(err)
