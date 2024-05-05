import pandas as pd
import re
from collections import Counter

def find_preceding_words(data_frame, node_types):
    # Combine node types into a regex pattern
    pattern = re.compile(r'\b([a-z]+)\s+(' + '|'.join(node_types) + r')\b')
     
    # List to store all preceding words
    all_preceding_words = []

    # Iterate over each text entry
    for text in data_frame['node_context']:
        # Find all matches of the pattern
        matches = pattern.findall(text)
        # Extract and collect all preceding words
        all_preceding_words.extend([match[0] for match in matches])

    return Counter(all_preceding_words)

# Load your data
df = pd.read_csv('data/context_of_disease_which_has_relation_to_genes.csv')

# Specify your node types
node_types = ["Anatomy", "BiologicalProcess", "Blend", "CellLine", "CellType", "CellularComponent", 
              "ClinicalLab", "Complex", "Compound", "Chromosome", "Cytoband", "DietarySupplement", 
              "Disease", "EC", "Environment", "Food", "Gene", "Location", "Haplotype", 
              "MolecularFunction", "MiRNA", "Organism", "Pathway", "PanGene", "PharmacologicClass", 
              "Protein", "ProteinDomain", "ProteinFamily", "PwGroup", "Reaction", "SDoH", 
              "SideEffect", "Symptom", "Variant"]

# Call the function
preceding_words_count = find_preceding_words(df, node_types)

# Print the results
print(preceding_words_count)
