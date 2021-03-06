# Globals
import signal
from io_func.model_io import log
import sys

stop_requested = False
init = False

def init_signal_handler():
   def handler(x,y):
       log("SIGUSR1 SIGNAL RECEIVED")
       global stop_requested 
       stop_requested = True
   
   global init      
   log("SETUP SIGNAL HANDLER SIGUSR1")
   signal.signal(signal.SIGUSR1, handler)
   init = True

def stop_if_stop_is_requested(): 
   global init
   global stop_requested
   if init == False:
    init_signal_handler()
   if stop_requested: 
        log("STOP REQUESTED")
        sys.exit(1)
   return

