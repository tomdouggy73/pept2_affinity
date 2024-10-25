#%%

# This script generates smiles strings for all natural di- and tri-peptides

# amino acid definitions in appendable format

smiles_fragments = {
        "Ala":"N[C@@]([H])(C)C(=O)",
        "Val":"N[C@@]([H])(C(C)C)C(=O)",
        "Ile":"N[C@@]([H])([C@]([H])(CC)C)C(=O)",
        "Leu":"N[C@@]([H])(CC(C)C)C(=O)",
        "Met":"N[C@@]([H])(CCSC)C(=O)",
        "Phe":"N[C@@]([H])(Cc1ccccc1)C(=O)",
        "Tyr":"N[C@@]([H])(Cc1ccc(O)cc1)C(=O)",
        "Trp":"N[C@@]([H])(CC(=CN2)C1=C2C=CC=C1)C(=O)",
        "Ser":"N[C@@]([H])(CO)C(=O)",
        "Thr":"N[C@@]([H])([C@]([H])(O)C)C(=O)",
        "Asn":"N[C@@]([H])(CC(=O)N)C(=O)",
        "Gln":"N[C@@]([H])(CCC(=O)N)C(=O)",
        "Cys":"N[C@@]([H])(CS)C(=O)",
        "Gly":"NCC(=O)",
        "Sar":"N(C)CC(=O)",
        "Pro":"N1[C@@]([H])(CCC1)C(=O)",
        "Hpr":"N1[C@@]([H])(C[C@@](O)C1)C(=O)",
        "Pip":"N1[C@@]([H])(CCCC1)C(=O)",
        "Arg":"N[C@@]([H])(CCCNC(=[NH2+])N)C(=O)",
        "His":"N[C@@]([H])(CC1=CN=C-N1)C(=O)",
        "Lys":"N[C@@]([H])(CCCC[NH3+])C(=O)",
        "Orn":"N[C@@]([H])(CCC[NH3+])C(=O)",
        "Dab":"N[C@@]([H])(CC[NH3+])C(=O)",
        "Nle":"N[C@@]([H])(CCCC)C(=O)",
        "Asp":"N[C@@]([H])(CC(=O)[O-])C(=O)",
        "Glu":"N[C@@]([H])(CCC(=O)[O-])C(=O)",
        "D-Ala":"N[C@]([H])(C)C(=O)",
        "D-Val":"N[C@]([H])(C(C)C)C(=O)",
        "D-Ile":"N[C@]([H])([C@]([H])(CC)C)C(=O)",
        "D-Leu":"N[C@]([H])(CC(C)C)C(=O)",
        "D-Met":"N[C@]([H])(CCSC)C(=O)",
        "D-Phe":"N[C@]([H])(Cc1ccccc1)C(=O)",
        "D-Tyr":"N[C@]([H])(Cc1ccc(O)cc1)C(=O)",
        "D-Trp":"N[C@]([H])(CC(=CN2)C1=C2C=CC=C1)C(=O)",
        "D-Ser":"N[C@]([H])(CO)C(=O)",
        "D-Thr":"N[C@]([H])([C@]([H])(O)C)C(=O)",
        "D-Asn":"N[C@]([H])(CC(=O)N)C(=O)",
        "D-Gln":"N[C@]([H])(CCC(=O)N)C(=O)",
        "D-Cys":"N[C@]([H])(CS)C(=O)",
        "D-Gly":"NCC(=O)",
        "D-Pro":"N1[C@]([H])(CCC1)C(=O)",
        "D-Arg":"N[C@]([H])(CCCNC(=[NH2+])N)C(=O)",
        "D-His":"N[C@]([H])(CC1=CN=C-N1)C(=O)",
        "D-Lys":"N[C@]([H])(CCCC[N])C(=O)",
        "D-Orn":"N[C@]([H])(CCC[N])C(=O)",
        "D-Dab":"N[C@]([H])(CC[N])C(=O)",
        "D-Asp":"N[C@]([H])(CC(=O)[O-])C(=O)",
        "D-Glu":"N[C@]([H])(CCC(=O)[O-])C(=O)"
}


# construct all the possible combinations of two and three of those fragments

def append_new_combinations_to_list(smiles_list,fragments_to_append):
    output = smiles_list.copy()
    for s1 in smiles_list:
        for s2 in fragments_to_append:
            output.append(s1+s2)
    return output

fragments = list(smiles_fragments.values())

all_smiles = fragments.copy()

all_smiles = append_new_combinations_to_list(all_smiles,fragments)
all_smiles = append_new_combinations_to_list(all_smiles,fragments)

all_smiles_prot = []

for s in all_smiles:
    s_out = s + "[O-]\n"
    
    if s.startswith(smiles_fragments['Pro']) or s.startswith(smiles_fragments['Sar']) or s.startswith(smiles_fragments['D-Pro']) or s.startswith(smiles_fragments['Hpr']) or s.startswith(smiles_fragments['Pip']):
        s_out = "[NH2+]" + s_out[1:]
    else:
        s_out = "[NH3+]" + s_out[1:]
    
    all_smiles_prot.append(s_out)

with open("all_peptide_smiles.dat", 'w') as f:
    f.writelines(all_smiles_prot)

print(all_smiles_prot)
# protonate the termini
# %%
