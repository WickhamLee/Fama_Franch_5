import fund_mgm_utilities as fmu
import numpy as np
import pandas as pd
import os
import random
import time
import warnings
import uuid


# ------------------------------------------------------------------
# Create a temporary bash file, execute a script and then delete it
# ------------------------------------------------------------------
def exec_bat(script, temp_folder, auto_clean = True):
    
    file_name = fmu.out_hash_value(fmu.now() + str(random.random) + str(uuid.uuid1()))
    
    file_path = temp_folder + "\\" + file_name + ".bat"
    out_file = open(file_path, "w")
    if auto_clean:
        out_file.write(script + " " + file_path.replace(".bat", ".verify"))
    else:
        out_file.write(script)
        
    out_file.close()
    
    os.startfile(file_path)

    verified = False
    if auto_clean:
        starttime = time.time()
        check_file_name = file_name + '.verify'
        check_file_path = file_path.replace('.bat', '.verify')
        verified = False
        while True and time.time() - starttime <= 60:
            if check_file_name in os.listdir(temp_folder):
                 os.remove(file_path)
                 try:
                     print(fmu.now() + ': Trying to delete ' + check_file_path)
                     os.remove(check_file_path)
                     verified = True
                 except:
                     verified = False
                 break
         
        if not(verified):
            warnings.warn('Execution not verified, expected message ' + file_path.replace(".bat", ".verify"))
         
    return verified

# ------------------------------------------------------------------
# Create a temporary bash file, execute a script and then delete it
# ------------------------------------------------------------------
def verify_exec_bat():
    return "import sys \nif(len(sys.argv))>= 2: \n\tif sys.argv[-1][-7:]=='.verify':\n\t\tout_file = open(sys.argv[-1], 'w') \n\t\tout_file.close()"





