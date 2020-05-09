import os
import time
import keyboard
import numpy as np
from os import listdir
import sys



def record_game():
    presses = np.array([])
    try:
        n = sys.argv[1]
    finally:
        print ('Enter the game number when running this script')
        print ('Do not skip any numbers and start at 1')
        print ('example: sudo python GetData.py 1')
    os.system('rm -rf GamePics/game{}'.format(n))
    os.system('mkdir GamePics/game{}'.format(n))
    i = 0

    time.sleep(2.5)
    print ('GO!')
    s = time.time()
    try:
        while time.time() - s < 200:
            os.system("screencapture -R60,125,600,150 GamePics/game{}/{}.png".format(n, i))
            if keyboard.is_pressed('up'):
                presses[-1] = 1
                presses = np.append(presses, [1])
            elif keyboard.is_pressed('down'):
                presses[-1]=2
                presses=np.append(presses,[2])
            else:
                presses = np.append(presses, [0])
            print (i)
            i += 1
    finally:
        presses = presses.reshape((-1, 1))
        print (presses.shape)
        np.save('labels/game{}.npy'.format(n), presses)

record_game()
