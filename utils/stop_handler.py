# Globals
import signal

stop_requested = False
init = False

def init_signal_handler():
   def handler(x,y):
       global stopRequested 
       stopRequested = True

   signal.signal(signal.SIGUSR1, handler)

def stop_if_stop_is_requested(): 
   if init == False:
    init_signal_handler()
   if stop_requested: 
        sys.exit(1)
   return

