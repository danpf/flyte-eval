#!/usr/bin/env python

from flytekit import dynamic, workflow

from flyte_science.workflow1.library.workflow import workflow1
from flyte_science.workflow2.library.workflow import workflow2
from flyte_science.workflow3.library.workflow import workflow3


@dynamic
def main() -> str:
    input_ = "TEST1 "
    wf1_ret = workflow1(input_=input_)
    wf2_ret = workflow2(input_=wf1_ret)
    wf3_ret = workflow3(input_=wf2_ret)
    return wf3_ret


if __name__ == "__main__":
    main()
