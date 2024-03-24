import logging
import os
from .singleton import singleton

class logger(singleton('logger', (object,), {})):
    def __init__(self):
        self.logDir_ = "logs"
        if not os.path.exists(self.logDir_):
            os.makedirs(self.logDir_)
        
        self.logFormat_ = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
        self.fileHandler_ = logging.FileHandler(os.path.join(self.logDir_, 'log'))
        self.fileHandler_.setFormatter(self.logFormat_)
        self.consoleHandler_ = logging.StreamHandler()
        self.consoleHandler_.setFormatter(self.logFormat_)

        self.logger_ = logging.getLogger('AppLoger')
        self.logger_.setLevel(logging.INFO)
        self.logger_.addHandler(self.fileHandler_)
        self.logger_.addHandler(self.consoleHandler_)
    
    def d(self, *args, **kwargs):
        self.logger_.debug(*args, **kwargs)

    def i(self, *args, **kwargs):
        self.logger_.info(*args, **kwargs)

    def w(self, *args, **kwargs):
        self.logger_.warning(*args, **kwargs)

    def e(self, *args, **kwargs):
        self.logger_.error(*args, **kwargs)

def d(*args, **kwargs):
    log = logger()
    log.d(*args, **kwargs)

def i(*args, **kwargs):
    log = logger()
    log.i(*args, **kwargs)

def w(*args, **kwargs):
    log = logger()
    log.w(*args, **kwargs)

def e(*args, **kwargs):
    log = logger()
    log.e(*args, **kwargs)