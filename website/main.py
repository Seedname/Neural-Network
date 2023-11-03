import winsound
from time import sleep
from random import randint

def beep (f, ms):
    winsound.Beep(f, ms)

for i in range(40):
    beep(15000, 500)
    sleep(randint(1, 2))