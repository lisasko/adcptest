"""
Created on Sep 28, 2017

@author: gpetrochenkov
"""
import threading


class MultiThread(threading.Thread):
    
    def __init__(self, thread_id, function, args=None):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.function = function
        self.args = args
        
    def run(self):
        
        if self.args is not None:
            self.function(**self.args)
        else:
            self.function()
