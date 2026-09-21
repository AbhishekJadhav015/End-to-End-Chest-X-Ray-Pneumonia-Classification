import sys

def error_message_detail(error , error_details : sys):
    "This function returns the error message with details or error secha as file name line number"
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script namne[{0} line number [{1}] error number [{2}]".format(file_name,exc_tb.tb_lineno , str(error))
    return error_message

class CustomException (Exception):
    "Custom Exception class which will be used to handle exception in the project"
    def __init__(self, error_message , error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_details=error_details)
        
    def __str__(self):
        return self.error_message