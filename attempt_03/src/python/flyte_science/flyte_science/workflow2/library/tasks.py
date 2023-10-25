from typing import TYPE_CHECKING
from flytekit import dynamic, task, workflow
from . import foo2_image_spec

if TYPE_CHECKING:
    from foo import FOO


@task(container_image=foo2_image_spec)
def task1() -> "Foo":
    from foo import Foo

    return Foo()


@task(container_image=foo2_image_spec)
def combine_task_outputs(input_og: str, input_foo: "Foo") -> str:
    return f"{input_og} -> {str(input_foo.do_v2_thing())}"


@workflow(container_image=foo2_image_spec)
def task2(input_: str) -> str:
    return combine_task_outputs(input_og=input_, input_foo=task1())
