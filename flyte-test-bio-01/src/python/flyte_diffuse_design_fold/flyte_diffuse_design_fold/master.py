from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin
from flytekit import dynamic, task
from flytekit.types.file import FlyteFile

from .esmfold_workflow import run_esmfold
from .foldingdiff_wf import run_foldingdiff
from .mpnn_workflow import ProteinMPNNOutput, run_protein_mpnn


@dataclass
class AnalyzedDiffusionResult(DataClassJsonMixin):
    sequence: str
    folded_pdb: FlyteFile
    diffused_pdb: FlyteFile
    rmsd: float


@task
def flatten_list_of_lists_of_strings(input_lol: list[list[str]]) -> list[str]:
    results = [y for x in input_lol for y in x]
    return results


@task
def flatten_list_of_lists_mpnn_outputs(input_lol: list[list[ProteinMPNNOutput]]) -> list[ProteinMPNNOutput]:
    results = [y for x in input_lol for y in x]
    return results


@task
def get_sequences_from_protein_mpnn_outputs(input_lol: list[ProteinMPNNOutput]) -> list[str]:
    return [x.sequence for x in input_lol]


@task
def flatten_list_of_lists_of_flytefiles(input_lol: list[list[FlyteFile]]) -> list[FlyteFile]:
    results = [y for x in input_lol for y in x]
    return results


@dynamic
def run_mpnn(
    input_pdb_files: list[FlyteFile],
    protein_mpnn_commandline_args: list[str],
    protein_mpnn_num_designs: int,
    useless_arg: int,
) -> list[ProteinMPNNOutput]:
    results = []
    for i, input_pdb_file in enumerate(input_pdb_files):
        results.append(
            run_protein_mpnn(
                input_pdb_file=input_pdb_file,
                protein_mpnn_commandline_args=protein_mpnn_commandline_args,
                protein_mpnn_num_designs=protein_mpnn_num_designs,
                useless_arg=[useless_arg, i],
            )
        )
    return flatten_list_of_lists_mpnn_outputs(input_lol=results)


@task(container_image="localhost:30000/biopython:latest", environment={"PYTHONPATH": "/root"})
def compare_predicted_backbones_to_folded_sequences(
    backbone_pdbs: list[FlyteFile], folded_pdbs: list[FlyteFile], mpnn_results: list[ProteinMPNNOutput]
) -> list[AnalyzedDiffusionResult]:
    from io import StringIO

    from Bio.PDB import PDBParser, Superimposer

    def parse_pdb_string(pdb_string):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("temp", StringIO(pdb_string))
        return structure

    def filter_ca_atoms(structure):
        # Filter and return only the CA atoms from the structure
        ca_atoms = [atom for atom in structure.get_atoms() if atom.id == "CA"]
        return ca_atoms

    def calculate_rmsd(pdb_string1: str, pdb_string2: str) -> float:
        # Parse the PDB strings to get structure objects
        structure1 = parse_pdb_string(pdb_string1)
        structure2 = parse_pdb_string(pdb_string2)

        # Extract the first model and get the list of atoms for both structures
        atoms1 = filter_ca_atoms(structure1)
        atoms2 = filter_ca_atoms(structure2)

        # Ensure both structures have the same number of atoms
        if len(atoms1) != len(atoms2):
            raise ValueError("Both structures should have the same number of atoms for RMSD calculation.")

        # Initialize the Superimposer object and set the atom lists
        super_imposer = Superimposer()
        super_imposer.set_atoms(atoms1, atoms2)

        # Perform the alignment
        super_imposer.apply(atoms2)

        # Get the RMSD
        rmsd = super_imposer.rms
        return rmsd

    ret: list[AnalyzedDiffusionResult] = []
    if len(set(map(len, [backbone_pdbs, folded_pdbs, mpnn_results]))) != 1:
        raise RuntimeError(
            f"BAD LENGTHS backbone={len(backbone_pdbs)} folded={len(folded_pdbs)} mpnn={len(mpnn_results)}"
        )
    for backbone_pdb, folded_pdb, mpnn_result in zip(backbone_pdbs, folded_pdbs, mpnn_results):
        print(backbone_pdb, folded_pdb)
        with backbone_pdb.open("r") as fh:
            backbone_pdb_str = fh.read()
        with folded_pdb.open("r") as fh:
            folded_pdb_str = fh.read()
        rmsd = calculate_rmsd(backbone_pdb_str, folded_pdb_str)
        ret.append(
            AnalyzedDiffusionResult(
                sequence=mpnn_result.sequence, folded_pdb=folded_pdb, diffused_pdb=backbone_pdb, rmsd=rmsd
            )
        )
    ret.sort(key=lambda x: x.rmsd)
    return ret


@dynamic
def run_rf_mpnn_esm_wf(
    input_pdb_file: FlyteFile,
    diffusion_num_designs: int,
    foldingdiff_commandline_args: list[str],
    protein_mpnn_num_designs: int,
    protein_mpnn_commandline_args: list[str],
) -> list[AnalyzedDiffusionResult]:
    all_mpnn_results = []
    all_diff_results = []

    # We do 10 at a time because each foldingdiff workflow creates 10 backbones
    for x in range(int(diffusion_num_designs / 10) + 1):
        c_rf_diffusion_results = run_foldingdiff(
            num_designs=diffusion_num_designs,
            length=64,
            foldingdiff_commandline_args=foldingdiff_commandline_args,
            useless_arg=x,
        )
        for x in range(protein_mpnn_num_designs):
            all_diff_results.append(c_rf_diffusion_results)
        all_mpnn_results.append(
            run_mpnn(
                input_pdb_files=c_rf_diffusion_results,
                protein_mpnn_commandline_args=protein_mpnn_commandline_args,
                protein_mpnn_num_designs=protein_mpnn_num_designs,
                useless_arg=x,
            )
        )
    flattened_diff_results = flatten_list_of_lists_of_flytefiles(input_lol=all_diff_results)
    flattened_mpnn_results = flatten_list_of_lists_mpnn_outputs(input_lol=all_mpnn_results)
    all_mpnn_sequeunces = get_sequences_from_protein_mpnn_outputs(input_lol=flattened_mpnn_results)
    esm_results = run_esmfold(sequences=all_mpnn_sequeunces)
    ret = compare_predicted_backbones_to_folded_sequences(
        backbone_pdbs=flattened_diff_results,
        folded_pdbs=esm_results,
        mpnn_results=flattened_mpnn_results,
    )

    # Ideally we would like manually inspect the results of the top ~20 or so for this.
    # So, downloading pdb files here would be ideal.
    return ret
