# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:27:49 2020

@author: Russell J. Adams
"""

class memory:
    def __init__(self, max_mem_size=10000):
        self.memory = {}
        self.max_mem = max_mem_size
        self.mem_counter = 0
    
    def push(self, key):
        if self.mem_counter < self.max_mem:
            if not self.memory.get(key):
                self.memory[key] = key
                self.mem_counter += 1
    
    def get(self, key):
        return self.memory.get(key, 1)