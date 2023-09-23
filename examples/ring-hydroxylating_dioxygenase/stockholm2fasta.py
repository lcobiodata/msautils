from Bio import AlignIO

# Open the Stockholm format file
with open('PF00848-full.sto') as handle:
    # Parse the alignment
    alignment = AlignIO.read(handle, 'stockholm')

# Save the alignment in FASTA format
output_file = 'PF00848_full.txt'
AlignIO.write(alignment, output_file, 'fasta')

print(f"Stockholm alignment converted to FASTA format and saved to {output_file}.")
