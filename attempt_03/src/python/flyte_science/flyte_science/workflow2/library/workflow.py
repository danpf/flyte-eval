from flytekit import workflow


@workflow
def workflow2(input_: str) -> str:
    from .tasks import task2

    return task2(input_=input_)
