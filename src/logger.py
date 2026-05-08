import os
import logging
from datetime import datetime

LOG_FILE = f'{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log'

path = os.path.join(os.getcwd(), 'logs')
os.makedirs(path, exist_ok = True)

LOG = os.path.join(path, LOG_FILE)

logging.basicConfig(
    level = logging.INFO,
    format = '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    filename = LOG
)

logging.getLogger(__name__)