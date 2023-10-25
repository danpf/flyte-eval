#!/usr/bin/env python

import argparse
from typing import List
import subprocess
import sys
import logging

log = logging.getLogger(__name__)


def parseargs(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thing1", help="thing1")
    return parser.parse_args(args)

def main(_args: List[str]) -> None:
    args = parseargs(_args)
    
    for env in range(1, 4):
        script = f"""
        . ~/.zshrc
        micromamba create --name tflyte-subwf-0{env} -c conda-forge python=3.11 -y
        micromamba activate tflyte-subwf-0{env}
        pip install -e ../foo_v{env} --config-settings editable_mode=strict
        pip install -e ./flyte_science/workflow{env} --config-settings editable_mode=strict
        """
        with open("runme.sh", "w") as fh:
            fh.write(script)
        subprocess.call("zsh ./runme.sh", shell=True)

if __name__ == "__main__":
    main(sys.argv[1:])
