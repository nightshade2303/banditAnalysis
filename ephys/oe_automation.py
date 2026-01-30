# From : https://groups.google.com/g/open-ephys/c/Pre2Mzj1maE/m/gIYnyXnBAwAJ
# Using the open-ephys-python-tools you could set up the following script to automate your experiments:
from open_ephys import control

gui = control.OpenEphysHTTPServer()

record_node_id = ... #usually 101
new_directory = ... #in this case L:/4portProb_ephys/Box1_ephys/Bayleef/Ephys/


#Load your signal chain 

gui.load(“your_signal_chain.xml”)

 

#Change the save directory

gui.set_record_path(record_node_id, new_directory)

#Select recorded channels (first 4 channels from stream 0) what the hell is this?

gui.config(record_node_id, “SELECT 0 1 2 3 4”) 

#Start recording

gui.record(duration=3600) #records for 1 hr

 

gui.idle()

 

#Setup and run the next experiment

gui.set_record_path(record_node_id, new_directory_2)

gui.config(record_node_id, “SELECT 0 5 6 7 8”)

gui.record(duration=1800) # records for 30 min

gui.idle()

 

gui.quit()             

 

# You can find the rest of the currently available GUI remote controls here:

 

# https://github.com/open-ephys/open-ephys-python-tools/blob/main/src/open_ephys/control/http_server.py

 

# You can save/load config files via the GUI using the File-> Save/Open option, same effect as using the gui.load() command above. 