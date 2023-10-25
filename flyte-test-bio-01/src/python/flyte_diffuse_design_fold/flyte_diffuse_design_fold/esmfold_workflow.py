from pathlib import Path
from flytekit import task, Resources
from flytekit.types.file import FlyteFile


def convert_outputs_to_pdb(outputs):
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


@task(container_image="localhost:30000/esmfold:latest", environment={"PYTHONPATH": "/root"}, requests=Resources(cpu="7", mem="28Gi"))
def run_esmfold(sequences: list[str]) -> list[FlyteFile]:
    """
    This output should maybe be a FlyteFile instead?
    """
    from transformers import AutoTokenizer, EsmForProteinFolding
    import torch
    unique_seqs = list(sorted(set(sequences)))

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    # model = model.cuda()
    # model.esm = model.esm.half()
    model.trunk.set_chunk_size(64)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenized_input = tokenizer(unique_seqs, return_tensors="pt", add_special_tokens=False)['input_ids']
    # tokenized_input = tokenized_input.cuda()
    
    with torch.no_grad():
        output = model(tokenized_input)

    pdbs = convert_outputs_to_pdb(output)
    pdb_files = []
    for i, _pdb in enumerate(pdbs):
        pdb_file = f"{i:03}.pdb"
        with open(pdb_file, "w") as fh:
            fh.write(_pdb)
        pdb_files.append(FlyteFile(path=pdb_file))

    # This is just a little hack to speed up testing when you get lots of poly-G
    ret_pdbs = []
    for sequence in sequences:
        ret_pdbs.append(pdb_files[unique_seqs.index(sequence)])
    return ret_pdbs
