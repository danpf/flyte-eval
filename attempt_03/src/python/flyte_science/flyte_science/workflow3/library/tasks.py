from typing import TYPE_CHECKING
from flytekit import dynamic, task, workflow
from . import foo3_image_spec

if TYPE_CHECKING:
    from foo import Foo3


@task(container_image=foo3_image_spec)
def task1() -> "Foo3":
    from foo import Foo3
    return Foo3()


@task(container_image=foo3_image_spec)
def combine_task_outputs(input_og: str, input_foo: "Foo3") -> str:
    return f"{input_og} -> {str(input_foo.do_v3_thing())}"


@workflow(container_image=foo3_image_spec)
def task2(input_: str) -> str:
    return combine_task_outputs(input_og=input_, input_foo=task1())
