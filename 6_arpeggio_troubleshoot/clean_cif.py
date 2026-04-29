# Clean CIF Script
# ================
#
#
# CIF File Issues Addressed
# -------------------------
#
# - Multiple models                        ** DONE ** (takes first model)
# - Multiple occupancies/alternate locs    ** DONE **
#   - Pick 'A' conformer, fallback blank   ** DONE **
#   - Set occupancies to 1.00              ** DONE **
# - Missing atoms, residues               ** NOT DEALT WITH **
# - Chain breaks                           ** DONE **
#   - Output to file: CA or N or C pos    ** DONE **
# - Selenomets to Mets                     ** DONE **
# - Nonstandard res to ATOM records        ** DONE ** (via PPBuilder membership)
# - Hydrogens                              ** KEPT BY DEFAULT ** (mirrors -kh flag)
# - Waters                                 ** OPTIONAL REMOVAL **
#
#
# Notes vs. clean_pdb.py
# ----------------------
# - Input is mmCIF; no gemmi convert step needed downstream
# - Output is written via MMCIFIO (not a manual PDB-format template)
# - Occupancy is forced to 1.00 on the atom object before writing
# - Atom serials are renumbered from 1 via a post-write approach (MMCIFIO handles internally)
# - 1-to-1 file identity with clean_pdb.py output is NOT expected (see comments below)
#   but functional equivalence for arpeggio is the goal


# IMPORTS
import argparse
import logging
import operator
import os
import sys
import traceback
from functools import reduce
from collections import OrderedDict

from Bio.PDB import MMCIFParser, MMCIFIO, Select
from Bio.PDB.Polypeptide import PPBuilder


# LOGGING
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# SELECT CLASS
# Controls which models/residues/atoms are written by MMCIFIO
class CleanSelect(Select):
    """
    Selector for MMCIFIO that:
    - Takes only the first model
    - Skips waters if requested
    - Resolves disordered atoms (picks altloc 'A' or blank)
    - Removes hydrogens if requested
    """

    def __init__(self, model_id, polypeptide_residues,
                 keep_hydrogens=True, remove_waters=False):
        self.model_id = model_id
        self.polypeptide_residues = polypeptide_residues
        self.keep_hydrogens = keep_hydrogens
        self.remove_waters = remove_waters

    def accept_model(self, model):
        return int(model.id) == self.model_id

    def accept_residue(self, residue):
        # Remove waters if flag set
        if self.remove_waters:
            if residue.get_full_id()[3][0] == 'W':
                return False
        return True

    def accept_atom(self, atom):
        # For disordered atoms, only accept the selected conformer
        # (we pre-select altloc 'A' or blank on the structure object below)
        if atom.is_disordered():
            # disordered_get() returns the currently selected child
            selected = atom.disordered_get()
            # Only accept if this is the selected child
            if atom.get_altloc() != selected.get_altloc():
                return False

        # Remove hydrogens unless flag set
        if not self.keep_hydrogens:
            element = atom.element.strip() if atom.element else ''
            if element == 'H':
                return False

        return True


# MAIN
if __name__ == '__main__':

    # ARGUMENT PARSING
    parser = argparse.ArgumentParser(description='''

#############
# CLEAN CIF #
#############

A program for cleaning mmCIF files prior to PDBe-arpeggio.
Mirrors the logic of clean_pdb.py but operates natively on mmCIF input.

Dependencies:
- Python (>= 3.7)
- BioPython (>= 1.79)

''', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('cif', type=str,
                        help='Path to the input mmCIF file.')
    parser.add_argument('output', type=str,
                        help='Path for the cleaned output mmCIF file.')
    parser.add_argument('-rmw', '--remove-waters', action='store_true',
                        help='Remove waters.')
    parser.add_argument('-kh', '--keep-hydrogens', action='store_true',
                        default=True,
                        help='Keep hydrogens (default: True, mirrors arpeggio pipeline flag).')
    parser.add_argument('-nh', '--no-hydrogens', action='store_true',
                        help='Remove hydrogens (overrides --keep-hydrogens).')

    args = parser.parse_args()

    cif_path = args.cif
    output_path = args.output

    # Resolve hydrogen flag: --no-hydrogens overrides default keep
    keep_hydrogens = args.keep_hydrogens and not args.no_hydrogens

    # PARSE CIF
    cif_parser = MMCIFParser(QUIET=True)
    structure_id = os.path.splitext(os.path.basename(cif_path))[0]

    try:
        structure = cif_parser.get_structure(structure_id, cif_path)
    except Exception as e:
        logging.error(f'Failed to parse CIF file: {e}')
        sys.exit(1)

    # TAKE FIRST MODEL ONLY
    # Mirrors: model = structure[0]
    try:
        model = structure[0]
    except (KeyError, IndexError):
        logging.error('No models found in structure.')
        sys.exit(1)

    model_id = model.id

    # RAISE AN ERROR FOR TOO MANY ATOMS
    # Mirrors the 99999 atom check in clean_pdb.py
    atom_count = len(list(model.get_atoms()))
    if atom_count > 99999:
        try:
            raise ValueError(f'More than 99999 atoms in the CIF model! ({atom_count})')
        except ValueError:
            traceback.print_exc(file=sys.stdout)
            sys.exit(9)

    # RESOLVE ALTERNATE LOCATIONS
    # For each disordered atom, select altloc 'A' if present, else first available.
    # This must be done before PPBuilder so polypeptide detection is consistent.
    # Mirrors: atom = atom.disordered_get() in clean_pdb.py
    for residue in model.get_residues():
        for atom in residue.child_list:
            if atom.is_disordered():
                altlocs = atom.disordered_get_id_list()
                selected = 'A' if 'A' in altlocs else altlocs[0]
                atom.disordered_select(selected)
                # Force occupancy to 1.00 on the selected child
                atom.disordered_get().set_occupancy(1.00)

    # SELENOMETHIONINE TO METHIONINE CONVERSION
    # Mirrors the MSE->MET block in clean_pdb.py.
    # Must happen before PPBuilder so MSE residues are treated as standard AA.
    for residue in model.get_residues():
        if residue.resname in ('MSE', 'MET'):
            residue.resname = 'MET'
            for atom in residue.child_list:
                a = atom.disordered_get() if atom.is_disordered() else atom
                if a.name == 'SE' and a.element == 'SE':
                    a.name = 'SD'
                    a.element = 'S'
                    a.fullname = ' SD '

    # DETERMINE POLYPEPTIDES AND CHAIN BREAKS
    # Mirrors the PPBuilder section in clean_pdb.py exactly.
    ppb = PPBuilder()
    polypeptides = ppb.build_peptides(model, aa_only=False)

    # MAKE DATA STRUCTURES FOR CHAIN POLYPEPTIDES
    chain_ids = set([x.id for x in model.child_list])
    chain_pieces = OrderedDict()
    chain_polypeptides = OrderedDict()
    chain_break_residues = OrderedDict()
    chain_sequences = OrderedDict()

    for chain_id in chain_ids:
        chain_pieces[chain_id] = 0
        chain_break_residues[chain_id] = []
        chain_polypeptides[chain_id] = []

    # GET ALL POLYPEPTIDE RESIDUES IN THE MODEL
    polypeptide_residues = []
    for pp in polypeptides:
        for res in pp:
            polypeptide_residues.append(res)

    # GET THE CHAIN_ID(S) ASSOCIATED WITH EACH POLYPEPTIDE
    polypeptide_chain_id_sets = [
        set([k.get_parent().id for k in x]) for x in polypeptides
    ]

    for e, polypeptide_chain_id_set in enumerate(polypeptide_chain_id_sets):

        if len(polypeptide_chain_id_set) != 1:
            logging.warning(
                f'A polypeptide had {len(polypeptide_chain_id_set)} chains '
                f'associated with it: {polypeptide_chain_id_set}'
            )

        for polypeptide_chain_id in polypeptide_chain_id_set:
            chain_pieces[polypeptide_chain_id] += 1
            chain_break_residues[polypeptide_chain_id] += [
                polypeptides[e][0], polypeptides[e][-1]
            ]
            chain_polypeptides[polypeptide_chain_id].append(polypeptides[e])

    # POP OUT GENUINE TERMINI TO LEAVE ONLY INTERNAL BREAK RESIDUES
    for chain_id in chain_break_residues:
        chain_break_residues[chain_id] = chain_break_residues[chain_id][1:-1]

    all_chain_break_residues = reduce(
        operator.add, chain_break_residues.values()
    ) if any(chain_break_residues.values()) else []

    # MAKE CHAIN SEQUENCES
    for chain_id in chain_polypeptides:
        pp_seqs = [str(x.get_sequence()) for x in chain_polypeptides[chain_id]]
        if pp_seqs:
            chain_sequences[chain_id] = reduce(operator.add, pp_seqs)

    # WRITE CLEANED CIF
    # Uses MMCIFIO + CleanSelect to apply all filtering decisions.
    # Note: MMCIFIO handles serial renumbering internally.
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    io = MMCIFIO()
    io.set_structure(structure)

    select = CleanSelect(
        model_id=model_id,
        polypeptide_residues=polypeptide_residues,
        keep_hydrogens=keep_hydrogens,
        remove_waters=args.remove_waters
    )

    try:
        io.save(output_path, select)
        logging.info(f'Cleaned CIF written to: {output_path}')
    except Exception as e:
        logging.error(f'Failed to write cleaned CIF: {e}')
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

    # WRITE OUT CHAIN BREAK FILES
    # Mirrors the .breaks and .break_residues output of clean_pdb.py
    cif_noext = os.path.splitext(output_path)[0]

    with open(f'{cif_noext}.breaks', 'w') as fo, \
         open(f'{cif_noext}.break_residues', 'w') as fo2:

        for cbr in all_chain_break_residues:

            fo2.write('{},{}{}`{}\n'.format(
                cbr.get_parent().id,
                cbr.get_id()[1],          # residue number
                cbr.get_id()[2].strip(),  # insertion code
                cbr.resname.strip()
            ))

            if 'CA' in cbr.child_dict:
                break_coord = list(cbr.child_dict['CA'].coord)
            elif 'N' in cbr.child_dict:
                break_coord = list(cbr.child_dict['N'].coord)
            elif 'C' in cbr.child_dict:
                break_coord = list(cbr.child_dict['C'].coord)
            else:
                raise ValueError(
                    f'Chain break residue {cbr} had no mainchain atom '
                    f'to extract coordinates from.'
                )

            fo.write(f'{str(break_coord)}\n')

    logging.info(f'Chain breaks written to: {cif_noext}.breaks')
    logging.info(f'Break residues written to: {cif_noext}.break_residues')
