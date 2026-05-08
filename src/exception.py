import sys
from src.logger import logging

def error_msg(error):
    _, _, exc_tb = sys.exc_info()

    filename = exc_tb.tb_frame.f_code.co_filename

    error_message = f'Error occurred at python script at {filename}, line no. {exc_tb.tb_lineno}, error: {str(error)}'

    return error_message

class CustomException(Exception):
    def __init__(self, error):
        super().__init__(error)
        self.error = error_msg(error)
        logging.error(self.error)

    def __str__(self):
        return self.error

