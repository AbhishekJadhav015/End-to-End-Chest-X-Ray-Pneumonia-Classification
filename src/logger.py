import logging 
import os
from datetime import datetime

# setup logging configuration
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log" 
log_dir_path = os.path.join(os.getcwd(),"logs")
#creating directory
os.makedirs(log_dir_path, exist_ok=True)
# Defining full absolute path
LOG_FILE_PATH = os.path.join(log_dir_path , LOG_FILE)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO ,
)