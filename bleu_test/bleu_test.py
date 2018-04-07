reference = 'Two person be in a small race car drive by a green hill . It is a beautiful morning .'
output = 'Two person in race uniform in a street car .It is a beautiful morning .'

#output = reference
with open('output', 'w') as output_file:
    output_file.write(reference)

with open('reference', 'w') as reference_file:
    reference_file.write(output)

from os import system
cmd = './multi-bleu.perl reference < output'
system(cmd)
#import subprocess
#out = subprocess.check_output(cmd)
#print(output)
import os
bleu_output = os.popen(cmd).readlines()
with open('bleu_scores.txt', 'w') as fout:
    for i in bleu_output:
    	fout.write(i)

