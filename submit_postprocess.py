import os
import time
import itertools
import numpy as np
from astropy.io import ascii

def main():

    # Read IDs and distances
    data = ascii.read("/n/regal/iacs/rafael/all-distances.txt")
    id_list = data['recno']

    for i in np.arange(len(id_list)):
    #for i in np.arange(0,3):    
        arg = id_list[i]
        arg_str = str(arg)
        task = ('sbatch -p seas_iacs -c 2 -t 1-00:00 --mem=16000 --wrap="python post_process.py {}"').format(arg_str)

        print(task)
        os.system(task)

if __name__ == '__main__':
    main()
