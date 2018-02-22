# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:05:39 2018

@author: SrivatsanPC
"""
import operator

class BeamSearch():
    def __init__(self, beam_size = 5, init_beam = 'dummy'):
        #Initialize beam with start of sentence token.
        self.beam = [init_beam]
        self.scores = []
        self.beam_size = beam_size
        self.terminate = False
            
    def perform_beam_search(self,): 
        children_score_map = {}
        while not terminate:
            if len(self.beam):                
                for parent in self.beam:
                    #IF PARENT HAS NOT REACHED <EOS>
                    children_score_map.update(self.get_children_with_score(parent))
                sorted_children_score_map = sorted(children_score_map.items(), 
                                                   key=operator.itemgetter(1), reverse = True)
                self.beam = list(sorted_children_score_map.keys())[:self.beam_size]  
            #TODO : ADD TERMINATION CRITERIA. 
            #IF EACH HYPOTHESES IN BEAM REACHED <eos>, TERMINATE BEAM              
                
            else:
                raise Exception("Beam cannot be empty")    
        
        ##Once terminated return top element of beam.
        return self.beam[0]
    
    def get_children_with_score(parent):
        #Return a single dictionary with key as a child( sequence of y_0.....y_t+1 assuming parent is
        #y_0 to y_t)
        #RUN FORWARD LOOP WITH PARENT AS HISTORY.
        
                
        
        