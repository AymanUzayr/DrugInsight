import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem
from torch_geometric.data import Data


# ── Atom features ──────────────────────────────────────────
def atom_features(atom):
    return [
        atom.GetAtomicNum(),                         # element
        atom.GetDegree(),                            # number of bonds
        atom.GetFormalCharge(),                      # charge
        int(atom.GetHybridization()),                # sp, sp2, sp3 etc
        int(atom.GetIsAromatic()),                   # aromatic ring
        atom.GetTotalNumHs(),                        # hydrogen count
        int(atom.IsInRing()),                        # in a ring
        atom.GetMass() / 100.0,                      # normalized mass
    ]

# ── Bond features ──────────────────────────────────────────
def bond_features(bond):
    bt = bond.GetBondType()
    return [
        int(bt == rdchem.BondType.SINGLE),
        int(bt == rdchem.BondType.DOUBLE),
        int(bt == rdchem.BondType.TRIPLE),
        int(bt == rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]

# ── SMILES → PyG graph ─────────────────────────────────────
def smiles_to_graph(smiles):
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    # Node features
    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float
    )

    # Edge index + edge features
    edge_index = []
    edge_attr  = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        edge_index += [[i, j], [j, i]]  # undirected
        edge_attr  += [bf, bf]

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ── Process list of SMILES ─────────────────────────────────
def process_smiles_list(smiles_list):

    graphs = []
    invalid = 0

    for s in smiles_list:

        g = smiles_to_graph(s)

        if g is None:
            invalid += 1
            continue

        graphs.append(g)

    print("Invalid SMILES skipped:", invalid)
    print("Valid graphs:", len(graphs))

    return graphs
# ── Test ───────────────────────────────────────────────────
if __name__ == '__main__':
    smiles = "OC1=CC=CC(=C1)C-1=C2\CCC(=N2)\C(=C2/N\C(\C=C2)=C(/C2=N/C(/C=C2)=C(\C2=CC=C\-1N2)C1=CC(O)=CC=C1)C1=CC(O)=CC=C1)\C1=CC(O)=CC=C1"  
    graph = process_smiles_list([smiles])
    if len(graph) == 0:
        print("No valid molecules")
    else:
        g = graph[0]

        print("Nodes:", g.x.shape)
        print("Edges:", g.edge_index.shape)
        print("Edge features:", g.edge_attr.shape)