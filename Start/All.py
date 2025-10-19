"""
Vérifie sequence
"""

sequence = "ATGCTAGCTAGGCTAATAG"
VALIDE=True
for s in sequence:
    if s != 'A' and s != 'T' and s != 'G' and s != 'C':

        VALIDE=False
if VALIDE:
    gc_content = (sequence.count("G") + sequence.count("C")) / len(sequence) * 100

else:
    pass
"""
Trouve le pourcentage de C et de G et renvoie la description de chaques séquences 
"""
from Bio import SeqIO

for record in SeqIO.parse("sequences.fasta", "fasta"):
    seq=record.seq
    #print(record.id)
    print(record.description)

    print(len(seq)//3)


