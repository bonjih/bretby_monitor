import re
import time
import pandas as pd


bret_data = []


def format_df():

    # for some reason could not split using [df[0].split(' ', expand=True)]
    if len(bret_data) != 0:
        data_all = [bret_data[i:i + 6] for i in range(0, len(bret_data), 6)]
        df = pd.DataFrame(data_all)
        df.rename(columns={0: 'Camera', 1: 'x', 2: 'y', 3: 't0', 4: 't1', 5: 'Area'}, inplace=True)
        return df
    else:
        pass


def format_data(i, cam_name, res, tup_1, tup_2, box_area, trail_history):
    bret_coords_all = []

    if not res:
        bret_coords_all.append(trail_history[i][0][0])
        bret_coords_all.append(cam_name)

        # add time of entry to detect if coords are changing rapidly
        bret_coords_all.append(time.time())

        t0 = bret_coords_all[2]
        t1 = bret_coords_all[-1]

        bret_coords_all.append(t1 - t0)

        spt_1 = [l.split(',(') for l in ' '.join(map(str, bret_coords_all)).split('(')]
        spt_2 = [l.split(')') for l in ' '.join(map(str, spt_1[-1])).split(',')]
        spt_3 = [l for l in re.split(r'(\s|\,)', spt_2[1][1].strip()) if l]

        x = spt_2[1][0][-3:]
        y = spt_2[0][-1]
        camera = spt_3[0]
        time_0 = spt_3[2]
        time_1 = spt_3[4]

        bret_data.append(camera)
        bret_data.append(x)
        bret_data.append(y)
        bret_data.append(time_0)
        bret_data.append(time_1)
        bret_data.append(box_area)
