import time
from pynput.mouse import Listener

file = open("mouse.txt","w+")
def get_millis():
    return int(time.time()*1000)
last_time = get_millis()

def on_move(x, y):
    global last_time
    file.write(str(x)+","+str(y)+","+str((get_millis()-last_time))+"\n")
    print(str(x)+","+str(y)+","+str((get_millis()-last_time))+"\n")
    last_time=get_millis();
    file.flush()

def on_click(x, y, button, pressed):
    global last_time
    file.write(str(pressed)+","+str(x) + "," + str(y) + "," + str((get_millis() - last_time))+"\n");
    print(str(pressed)+","+str(x) + "," + str(y) + "," + str((get_millis() - last_time))+"\n");
    file.flush()


# Collect events until released
with Listener(
        on_move=on_move,
        on_click=on_click,
       ) as listener:
    listener.join()