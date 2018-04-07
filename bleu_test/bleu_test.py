reference = 'Two person be in a small race car drive by a green hill .'
output = 'Two person in race uniform in a street car .'

with open('output', 'w') as output_file:
    output_file.write(reference)

with open('reference', 'w') as reference_file:
    reference_file.write(output)

from os import system
system('./multi-bleu.perl reference < output')