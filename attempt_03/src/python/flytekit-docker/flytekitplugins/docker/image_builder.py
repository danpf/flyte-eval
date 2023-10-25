import os
import pathlib
import shutil
import subprocess

import docker

from flytekit.configuration import DefaultImages
from flytekit.core import context_manager
from flytekit.core.constants import REQUIREMENTS_FILE_NAME
from flytekit.image_spec.image_spec import _F_IMG_ID, ImageBuildEngine, ImageSpec, ImageSpecBuilder


class DockerImageSpecBuilder(ImageSpecBuilder):
    """
    This class is used to build a docker image using envd.
    """

    def build_image(self, image_spec: ImageSpec):
        # pip_index == remote client
        if image_spec.pip_index:
            client = docker.DockerClient(base_url=image_spec.pip_index)
        else:
            client = docker.from_env()
        client.build()



        # cfg_path = create_envd_config(image_spec)
        #
        # if image_spec.registry_config:
        #     bootstrap_command = f"envd bootstrap --registry-config {image_spec.registry_config}"
        #     self.execute_command(bootstrap_command)
        #
        # build_command = f"envd build --path {pathlib.Path(cfg_path).parent}  --platform {image_spec.platform}"
        # if image_spec.registry:
        #     build_command += f" --output type=image,name={image_spec.image_name()},push=true"
        # self.execute_command(build_command)




ImageBuildEngine.register("docker", DockerImageSpecBuilder())
