# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 21:25:02 2018

@author: SrivatsanPC
"""
#USED FOR TESTING OUT RANDOM STUFF.
with open("LSTM_out_clean" + ".txt", "w") as fout:
    for i, l in enumerate(open("LSTM_output.txt"), 1):
       words = l.split()
       if words == []:
           continue
       else:
           #import pdb; pdb.set_trace()
           print(words[0]," ".join(words[1:]),file=fout)