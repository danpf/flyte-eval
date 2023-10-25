from flytekit import task, dynamic, map_task, workflow, Resources
from flytekit.types.file import FlyteFile

# TODO: input_pdb_file_text should probably be a FlyteFile
@task(container_image="localhost:30000/rfdiffusion:latest", environment={"PYTHONPATH": "/root"}, requests=Resources(mem="50Gi"))
def run_rf_diffusion(input_pdb_file_text: str, num_designs: int, rfdiffusion_commandline_args: list[str]) -> list[str]:
    """
    this is from https://github.com/RosettaCommons/RFdiffusion/blob/main/scripts/run_inference.py
    """
    import re
    import os, time, pickle
    import torch
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    import logging
    from rfdiffusion.util import writepdb_multi, writepdb
    from rfdiffusion.inference import utils as iu
    import numpy as np
    import random
    import glob
    from pathlib import Path

    output_directory = Path("/app/rf_files/outputs")
    output_directory.mkdir(exist_ok=True, parents=True)
    input_pdb_file_name = "my_input.pdb"
    with open(input_pdb_file_name, "w") as fh:
        fh.write(input_pdb_file_text)
    required_commandline_args = [
        f"inference.output_prefix={output_directory}/", 
        "inference.model_directory_path=/app/rf_files/models/",
        f"inference.input_pdb=/app/rf_files/inputs/{input_pdb_file_name}",
        f"inference.num_designs={num_designs}",
    ]

    def make_deterministic(seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    print("CWD:", os.getcwd())
    # config_path must be relative path?? stupid...
    initialize_config_dir(version_base=None, config_dir="/app/RFdiffusion/config/inference", job_name="test_app")
    conf = compose(config_name="base", overrides=rfdiffusion_commandline_args + required_commandline_args)

    log = logging.getLogger(__name__)
    if conf.inference.deterministic:
        make_deterministic()

    # Check for available GPU and print result of check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf)

    # Loop over number of designs to sample.
    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            print(e)
            m = re.match(".*_(\d+)\.pdb$", e)
            print(m)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1

    flyte_ret = []
    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        if conf.inference.deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        log.info(f"Making design {out_prefix}")
        if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
            )
            continue

        x_init, seq_init = sampler.sample_init()
        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        # Loop over number of reverse diffusion time steps.
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])  # remove singleton leading dimension

        # Flip order for better visualization in pymol
        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        denoised_xyz_stack = torch.flip(
            denoised_xyz_stack,
            [
                0,
            ],
        )
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        px0_xyz_stack = torch.flip(
            px0_xyz_stack,
            [
                0,
            ],
        )

        # For logging -- don't flip
        plddt_stack = torch.stack(plddt_stack)

        # Save outputs
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = seq_stack[-1]

        # Output glycines, except for motif region
        final_seq = torch.where(
            torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
        )  # 7 is glycine

        bfacts = torch.ones_like(final_seq.squeeze())
        # make bfact=0 for diffused coordinates
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
        # pX0 last step
        out = f"{out_prefix}.pdb"

        # Now don't output sidechains
        writepdb(
            out,
            denoised_xyz_stack[0, :, :4],
            final_seq,
            sampler.binderlen,
            chain_idx=sampler.chain_idx,
            bfacts=bfacts,
        )
        flyte_ret.append(out)

        # run metadata
        trb = dict(
            config=OmegaConf.to_container(sampler._conf, resolve=True),
            plddt=plddt_stack.cpu().numpy(),
            device=torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "CPU",
            time=time.time() - start_time,
        )
        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value
        with open(f"{out_prefix}.trb", "wb") as f_out:
            pickle.dump(trb, f_out)

        if sampler.inf_conf.write_trajectory:
            # trajectory pdbs
            traj_prefix = (
                os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
            )
            os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

            out = f"{traj_prefix}_Xt-1_traj.pdb"
            writepdb_multi(
                out,
                denoised_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )

            out = f"{traj_prefix}_pX0_traj.pdb"
            writepdb_multi(
                out,
                px0_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )

        log.info(f"Finished design in {(time.time()-start_time)/60:.2f} minutes")
    results = [open(x).read() for x in flyte_ret]
    return results

# @dynamic
# def _run_rf_diffusion_map_task(input_args: tuple[str, int, list[str]]) -> list[str]:
#     return run_rf_diffusion(input_pdb_file_text=input_args[0], num_designs=input_args[1], rfdiffusion_commandline_args=input_args[2])
#
# @task
# def combine_results(rfdiffusion_results: list[list[str]]) -> list[str]:
#     ret = []
#     for x in rfdiffusion_results:
#         ret += x
#     return ret
#
# @dynamic
# def run_rf_diffusion_map_task(input_pdb_file_text: str, num_designs: int, rfdiffusion_commandline_args: list[str]) -> list[str]:
#     to_submit = [(input_pdb_file_text, 10, rfdiffusion_commandline_args) for x in range(int(num_designs/10)+1)]
#     map_task_results = map_task(_run_rf_diffusion_map_task)(input_args=to_submit)
#     return combine_results(map_task_results)

