import json
import networkx as nx

# The 20 standard amino acids. Defined here (rather than only in the calling
# script) so every script that imports arpeggio_to_rin uses the same set.
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
}

# ---------------------------------------------------------------------------
# Interaction hierarchy (strongest -> weakest).
#
### this information can be found by calling "ring -h" with RING 4.0 installed
# RING's ranking:
#   SSBOND > PICATION > PIPISTACK > IONIC > HBOND > METAL_ION > PIHBOND
#          > HALOGEN > VDW > IAC
# translated into Arpeggio's contact vocabulary:
#   SSBOND    -> 'ssbond'        (pseudo-type: Arpeggio 'covalent' between two
#                                 CYS SG atoms; see _record_rank)
#   PICATION  -> 'cation_pi'
#   PIPISTACK -> 'aromatic'
#   IONIC     -> 'ionic'
#   HBOND     -> 'hbond'
#   METAL_ION -> 'metal_complex'
#   PIHBOND   -> 'donor_pi'
#   HALOGEN   -> 'xbond'
#   VDW       -> 'vdw'
#   IAC       -> 'proximal'
#
# Arpeggio types with NO RING equivalent are slotted between VDW and IAC
# (weak/polar/hydrophobic contacts) or at the bottom (clashes, and ordinary
# 'covalent' backbone bonds, which RING does not rank).
#
# Changing this list only changes which interaction is *reported* on an edge;
# it never changes which residue pairs are connected.
# ---------------------------------------------------------------------------
DEFAULT_CONTACT_HIERARCHY = [
    'ssbond',
    'cation_pi',
    'aromatic',
    'ionic',
    'hbond',
    'metal_complex',
    'donor_pi',
    'xbond',
    'vdw',
    # Arpeggio-only types (no RING equivalent)
    'weak_hbond',
    'polar',
    'weak_polar',
    'carbonyl',
    'hydrophobic',
]

# Contact types dropped before edges are built. A record whose contact types
# are all excluded creates no edge; a record with a mix keeps only its
# non-excluded types. Disulfides survive even though 'covalent' is excluded,
# because CYS SG-SG covalent contacts are relabeled 'ssbond' first.

# omit interatomic by omitting proximal setting in Arpeggio
DEFAULT_EXCLUDED_CONTACTS = ['proximal', 'vdw_clash', 'clash', 'covalent']

# allow interatomic
# DEFAULT_EXCLUDED_CONTACTS = ['vdw_clash', 'clash', 'covalent']

def _record_rank(contact, rank_lookup, unknown_rank, excluded,
                 bgn=None, end=None):
    """Best (lowest) hierarchy rank among the non-excluded contact types on
    one record, plus those types sorted strongest-first. Returns None if every
    type on the record is excluded.

    Arpeggio labels disulfides only as 'covalent', same as peptide bonds, so a
    'covalent' record between two CYS SG atoms is relabeled 'ssbond'."""
    types = list(contact) if isinstance(contact, list) else [contact]
    if (bgn is not None and end is not None and 'covalent' in types
            and bgn['label_comp_id'] == 'CYS' and end['label_comp_id'] == 'CYS'
            and bgn['auth_atom_id'] == 'SG' and end['auth_atom_id'] == 'SG'):
        types = ['ssbond' if t == 'covalent' else t for t in types]
    types = [t for t in types if t not in excluded]
    if not types:
        return None
    ranked = sorted(types, key=lambda t: rank_lookup.get(t, unknown_rank))
    best = rank_lookup.get(ranked[0], unknown_rank) if ranked else unknown_rank
    return best, ranked


def arpeggio_to_rin(json_file, pdb_id: str, label_lookup: dict,
                    contact_hierarchy=None, excluded_contacts=None):
    """Convert a single Arpeggio JSON to a NetworkX residue interaction graph.

    Only protein-protein interactions between the 20 standard amino acids are
    included. Waters, ligands, ions, and cofactors are excluded so that
    centrality measures reflect the protein interaction network only,
    consistent with Amitai et al. (2004).

    A residue pair is connected if at least one atom-atom contact survives the
    filters. When several contacts exist for the same pair, the one stored on
    the edge is chosen deterministically: the record whose strongest contact
    type ranks highest in `contact_hierarchy`, with ties broken by the
    shortest atom-atom distance (and then by atom names, so the result does
    not depend on JSON ordering).

    Each node receives a 'biolip_label' attribute with one of four values:
        "binding"   — residue contacts a ligand in BioLiP
        "catalytic" — residue is annotated as catalytic in BioLiP
        "dual"      — residue carries both binding and catalytic annotations
        "none"      — residue is absent from BioLiP for this structure

    Parameters
    ----------
    json_file         : path to the Arpeggio JSON
    pdb_id            : lowercase PDB ID (e.g. "1ac0"), used for label lookup
    label_lookup      : dict returned by load_biolip_labels()
    contact_hierarchy : list of Arpeggio contact types, strongest first.
                        Defaults to DEFAULT_CONTACT_HIERARCHY. Types not in
                        the list rank below every listed type.
    excluded_contacts : contact types to ignore entirely. Defaults to
                        DEFAULT_EXCLUDED_CONTACTS (proximal, vdw_clash, clash,
                        covalent). Records left with no contact types are
                        skipped and counted in n_filtered.
    """
    if contact_hierarchy is None:
        contact_hierarchy = DEFAULT_CONTACT_HIERARCHY
    if excluded_contacts is None:
        excluded_contacts = DEFAULT_EXCLUDED_CONTACTS
    excluded_contacts = set(excluded_contacts)
    rank_lookup  = {name: i for i, name in enumerate(contact_hierarchy)}
    unknown_rank = len(contact_hierarchy)

    with open(json_file, 'r') as f:
        interactions = json.load(f)

    rin = nx.Graph()
    residue_interactions = {}
    n_total    = 0
    n_filtered = 0

    for interaction in interactions:
        bgn = interaction['bgn']
        end = interaction['end']
        n_total += 1

        # ignore waters
        if bgn['label_comp_id'] == 'HOH' or end['label_comp_id'] == 'HOH':
            n_filtered += 1
            continue

        # ignore non-standard residues (ligands, ions, cofactors, modified AAs)
        if bgn['label_comp_id'] not in STANDARD_AA or end['label_comp_id'] not in STANDARD_AA:
            n_filtered += 1
            continue

        # ignore interactions with the same residue (self loop)
        if (bgn['auth_asym_id'] == end['auth_asym_id'] and
                bgn['auth_seq_id'] == end['auth_seq_id'] and
                bgn['pdbx_PDB_ins_code'] == end['pdbx_PDB_ins_code']):
            continue

        bgn_node = f"{bgn['auth_asym_id']}_{bgn['auth_seq_id']}"
        end_node = f"{end['auth_asym_id']}_{end['auth_seq_id']}"

        # Node IDs omit the insertion code, so residues like 52 and 52A share a
        # node ID. A contact between them passes the check above but would
        # become a self-loop here, so skip it.
        if bgn_node == end_node:
            continue

        res_pair = tuple(sorted([bgn_node, end_node]))

        # Hierarchy-based selection: keep the best-ranked contact per pair.
        ranked = _record_rank(interaction['contact'], rank_lookup,
                              unknown_rank, excluded_contacts, bgn, end)
        if ranked is None:  # only excluded contact types on this record
            n_filtered += 1
            continue
        rank, ranked_types = ranked
        sort_key = (rank,
                    interaction['distance'],
                    bgn['auth_atom_id'],
                    end['auth_atom_id'])

        if res_pair in residue_interactions:
            if sort_key >= residue_interactions[res_pair]['sort_key']:
                continue  # existing record is stronger (or equal): keep it

        # BioLiP label lookup — keyed by (pdb_id, chain, auth_seq_id as int)
        bgn_key   = (pdb_id, bgn['auth_asym_id'], int(bgn['auth_seq_id']))
        end_key   = (pdb_id, end['auth_asym_id'], int(end['auth_seq_id']))
        bgn_label = label_lookup.get(bgn_key, "none")
        end_label = label_lookup.get(end_key, "none")

        residue_interactions[res_pair] = {
            'sort_key': sort_key,
            'interaction': {
                'distance':     interaction['distance'],
                'contact_type': ranked_types,
                'bgn_atom':     bgn['auth_atom_id'],
                'end_atom':     end['auth_atom_id'],
                'bgn_res':      bgn['label_comp_id'],
                'end_res':      end['label_comp_id'],
            },
            'nodes': {
                bgn_node: {'residue_type': bgn['label_comp_id'],
                           'chain':        bgn['auth_asym_id'],
                           'residue_num':  bgn['auth_seq_id'],
                           'biolip_label': bgn_label},
                end_node: {'residue_type': end['label_comp_id'],
                           'chain':        end['auth_asym_id'],
                           'residue_num':  end['auth_seq_id'],
                           'biolip_label': end_label},
            }
        }

    for (node1, node2), data in residue_interactions.items():
        for node in (node1, node2):
            if not rin.has_node(node):
                rin.add_node(node, **data['nodes'][node])

        idet    = data['interaction']
        contact = ','.join(idet['contact_type'])

        rin.add_edge(
            node1, node2,
            distance=idet['distance'],
            contact_types=contact,
            bgn_atom=idet['bgn_atom'],
            end_atom=idet['end_atom'],
            bgn_res=idet['bgn_res'],
            end_res=idet['end_res'],
        )

    return rin, n_total, n_filtered


# ---------------------------------------------------------------------------
# Hydrogen-bond experiments
#
# Both runs use the same settings as arpeggio_to_rin (same hierarchy, same
# excluded contact types, same filters). The ONLY difference is which of the
# two hydrogen-bond types is removed:
#   hbond experiment      : 'weak_hbond' removed -> only 'hbond' counts
#   weak_hbond experiment : 'hbond' removed      -> only 'weak_hbond' counts
# All other contact types (ionic, aromatic, vdw, ...) are still counted.
#
# The type is removed from BOTH the exclusion set and the hierarchy list.
# Removing it from the hierarchy alone would not be enough: an unlisted type
# still creates edges (it just ranks last), so the exclusion set is what
# actually removes the edges.
# ---------------------------------------------------------------------------
def _experiment_settings(removed_type, contact_hierarchy, excluded_contacts):
    if contact_hierarchy is None:
        contact_hierarchy = DEFAULT_CONTACT_HIERARCHY
    if excluded_contacts is None:
        excluded_contacts = DEFAULT_EXCLUDED_CONTACTS
    hierarchy = [t for t in contact_hierarchy if t != removed_type]
    excluded  = sorted(set(excluded_contacts) | {removed_type})
    return hierarchy, excluded


def arpeggio_to_rin_hbond_exp(json_file, pdb_id: str, label_lookup: dict,
                              contact_hierarchy=None, excluded_contacts=None):
    """Experiment: count 'hbond' but NOT 'weak_hbond'. Everything else matches
    arpeggio_to_rin. Returns (rin, n_total, n_filtered)."""
    hierarchy, excluded = _experiment_settings('weak_hbond', contact_hierarchy,
                                               excluded_contacts)
    return arpeggio_to_rin(json_file, pdb_id, label_lookup,
                           contact_hierarchy=hierarchy,
                           excluded_contacts=excluded)

## will not use but keep here
def arpeggio_to_rin_weak_hbond_exp(json_file, pdb_id: str, label_lookup: dict,
                                   contact_hierarchy=None,
                                   excluded_contacts=None):
    """Experiment: count 'weak_hbond' but NOT 'hbond'. Everything else matches
    arpeggio_to_rin. Returns (rin, n_total, n_filtered)."""
    hierarchy, excluded = _experiment_settings('hbond', contact_hierarchy,
                                               excluded_contacts)
    return arpeggio_to_rin(json_file, pdb_id, label_lookup,
                           contact_hierarchy=hierarchy,
                           excluded_contacts=excluded)


if __name__ == "__main__":
    # ---- settings: edit here ------------------------------------------------
    json_file         = "example_arpeggio.json"
    pdb_id            = "1ac0"
    label_lookup      = {}      # replace with the dict from load_biolip_labels()
    contact_hierarchy = None    # None -> DEFAULT_CONTACT_HIERARCHY
    excluded_contacts = None    # None -> DEFAULT_EXCLUDED_CONTACTS
    # (STANDARD_AA must be defined, e.g. imported from your imports module.)
    # -------------------------------------------------------------------------

    for name, func in [("baseline",       arpeggio_to_rin),
                       ("hbond only",     arpeggio_to_rin_hbond_exp),
                       ("weak_hbond only", arpeggio_to_rin_weak_hbond_exp)]:
        rin, n_total, n_filtered = func(
            json_file, pdb_id, label_lookup,
            contact_hierarchy=contact_hierarchy,
            excluded_contacts=excluded_contacts)
        print(f"{name:16s} nodes={rin.number_of_nodes():5d} "
              f"edges={rin.number_of_edges():5d} "
              f"(records: {n_total}, filtered: {n_filtered})")
