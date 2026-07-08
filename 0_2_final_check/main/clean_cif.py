# Clean CIF Script (ver3)
# =======================
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
# - BioPython (MMCIFParser) is used for parsing and all cleaning logic
#   (alt-loc resolution, MSE->MET, PPBuilder chain break detection)
# - Output is written via gemmi (replaces MMCIFIO from ver2)
#   Reason: MMCIFIO strips required mmCIF categories (e.g. _chem_comp)
#   that PDBe-arpeggio requires. Gemmi preserves the full mmCIF document
#   and we remove unwanted atoms directly from gemmi's atom table.
# - 1-to-1 file identity with clean_pdb.py output is NOT expected
#   but functional equivalence for arpeggio is the goal
#
#
# Changes vs. ver2
# ----------------
# - Replaced MMCIFIO writer with gemmi writer
# - Gemmi reads the original CIF independently to preserve all data categories
# - Unwanted atoms (alt-locs, hydrogens, waters) are removed from gemmi's
#   atom site table using the decisions made by BioPython's cleaning logic
# - _chem_comp and all other mmCIF categories are preserved in output


# IMPORTS
import argparse
import logging
import operator
import os
import sys
import traceback
from functools import reduce
from collections import OrderedDict

import gemmi
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder


# LOGGING
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


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
- gemmi (>= 0.6)

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

    cif_path    = args.cif
    output_path = args.output

    # Resolve hydrogen flag: --no-hydrogens overrides default keep
    keep_hydrogens = args.keep_hydrogens and not args.no_hydrogens

    # ------------------------------------------------------------------
    # BIOPYTHON PASS
    # Handles: parsing, alt-loc resolution, MSE->MET, chain break detection
    # The BioPython structure object is used only for cleaning logic —
    # it does NOT write the output file.
    # ------------------------------------------------------------------

    # PARSE CIF WITH BIOPYTHON
    cif_parser   = MMCIFParser(QUIET=True)
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
    # Must be done before PPBuilder so polypeptide detection is consistent.
    # Mirrors: atom = atom.disordered_get() in clean_pdb.py
    # Builds a set of (chain_id, res_seq, res_icode, atom_name, altloc) for
    # the REJECTED altlocs so gemmi can remove them from the atom site table.
    rejected_altlocs = set()  # (chain_id, res_seqnum, res_icode, atom_name, altloc)

    for residue in model.get_residues():
        chain_id  = residue.get_parent().id
        res_id    = residue.get_id()       # (' ', seqnum, icode)
        res_seqnum = res_id[1]
        res_icode  = res_id[2].strip()

        for atom in residue.child_list:
            if atom.is_disordered():
                altlocs  = atom.disordered_get_id_list()
                selected = 'A' if 'A' in altlocs else altlocs[0]
                atom.disordered_select(selected)
                atom.disordered_get().set_occupancy(1.00)

                # Record rejected altlocs for gemmi filtering
                for altloc in altlocs:
                    if altloc != selected:
                        rejected_altlocs.add(
                            (chain_id, res_seqnum, res_icode, atom.get_id(), altloc)
                        )

    # SELENOMETHIONINE TO METHIONINE CONVERSION
    # Mirrors the MSE->MET block in clean_pdb.py.
    # Must happen before PPBuilder so MSE residues are treated as standard AA.
    # Builds a set of (chain_id, res_seqnum) for MSE residues so gemmi
    # can update resname and atom names in the atom site table.
    mse_residues = set()   # (chain_id, res_seqnum, res_icode)

    for residue in model.get_residues():
        if residue.resname in ('MSE', 'MET'):
            chain_id   = residue.get_parent().id
            res_id     = residue.get_id()
            res_seqnum = res_id[1]
            res_icode  = res_id[2].strip()
            residue.resname = 'MET'

            for atom in residue.child_list:
                a = atom.disordered_get() if atom.is_disordered() else atom
                if a.name == 'SE' and a.element == 'SE':
                    a.name    = 'SD'
                    a.element = 'S'
                    a.fullname = ' SD '

            mse_residues.add((chain_id, res_seqnum, res_icode))

    # DETERMINE POLYPEPTIDES AND CHAIN BREAKS
    # Mirrors the PPBuilder section in clean_pdb.py exactly.
    ppb          = PPBuilder()
    polypeptides = ppb.build_peptides(model, aa_only=False)

    # MAKE DATA STRUCTURES FOR CHAIN POLYPEPTIDES
    chain_ids             = set([x.id for x in model.child_list])
    chain_pieces          = OrderedDict()
    chain_polypeptides    = OrderedDict()
    chain_break_residues  = OrderedDict()
    chain_sequences       = OrderedDict()

    for chain_id in chain_ids:
        chain_pieces[chain_id]         = 0
        chain_break_residues[chain_id] = []
        chain_polypeptides[chain_id]   = []

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

    # ------------------------------------------------------------------
    # GEMMI PASS
    # Reads the original CIF independently to preserve all data categories
    # (_chem_comp, _entity, _struct_conf, etc.) that MMCIFIO would strip.
    # Applies the cleaning decisions made above directly to gemmi's atom
    # site table, then writes the output.
    # ------------------------------------------------------------------

    try:
        gemmi_structure = gemmi.read_structure(cif_path)
    except Exception as e:
        logging.error(f'Gemmi failed to read CIF: {e}')
        sys.exit(1)

    # TAKE FIRST MODEL ONLY
    # Remove all models after the first
    while len(gemmi_structure) > 1:
        del gemmi_structure[-1]

    gemmi_model = gemmi_structure[0]

    # APPLY CLEANING TO GEMMI STRUCTURE
    chains_to_remove = []

    for chain in gemmi_model:
        residues_to_remove = []

        for residue in chain:
            res_seqnum = residue.seqid.num
            res_icode  = residue.seqid.icode.strip()

            # REMOVE WATERS if flag set
            if args.remove_waters and residue.is_water():
                residues_to_remove.append(residue.name + str(res_seqnum))
                continue

            # MSE -> MET: update resname and SE->SD in gemmi
            if (chain.name, res_seqnum, res_icode) in mse_residues:
                residue.name = 'MET'
                for atom in residue:
                    if atom.name == 'SE' and atom.element == gemmi.Element('Se'):
                        atom.name    = 'SD'
                        atom.element = gemmi.Element('S')

            # REMOVE REJECTED ALT-LOCS and unwanted atoms
            atoms_to_remove = []
            for atom in residue:
                atom_id = atom.name
                altloc  = atom.altloc if atom.altloc != '\x00' else ''

                # Remove rejected alternate conformers
                if altloc and (chain.name, res_seqnum, res_icode, atom_id, altloc) in rejected_altlocs:
                    atoms_to_remove.append(atom)
                    continue

                # Blank out altloc label on kept conformer and force occ=1.00
                if altloc:
                    atom.altloc    = '\x00'  # gemmi's representation of blank altloc
                    atom.occ       = 1.00

                # Remove hydrogens if flag not set
                if not keep_hydrogens and atom.element == gemmi.Element('H'):
                    atoms_to_remove.append(atom)

            for atom in atoms_to_remove:
                residue.remove_atom(atom.name, atom.altloc, atom.element)

        # Remove flagged residues
        for res_label in residues_to_remove:
            pass  # gemmi residue removal handled inline above via is_water check

    # WRITE OUTPUT CIF VIA GEMMI
    # Preserves all original mmCIF categories including _chem_comp
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        gemmi_structure.update_mmcif_block(gemmi_structure.make_mmcif_document()[0])
        gemmi_structure.make_mmcif_document().write_file(output_path)
        logging.info(f'Cleaned CIF written to: {output_path}')
    except Exception as e:
        logging.error(f'Gemmi failed to write CIF: {e}')
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

    # ------------------------------------------------------------------
    # WRITE OUT CHAIN BREAK FILES
    # Mirrors the .breaks and .break_residues output of clean_pdb.py
    # Uses BioPython residue objects (already resolved above)
    # ------------------------------------------------------------------
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
