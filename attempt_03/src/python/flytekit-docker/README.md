# Flytekit Docker Plugin

With `flytekitplugins-docker`, people easily create a docker image using existing docker files.

To install the plugin, run the following command:

```bash
pip install flytekitplugins-docker
```

Example
```python
# from flytekit import task
# from flytekit.image_spec import ImageSpec
#
# @task(image_spec=ImageSpec(dockerfile="/Path/To/Dockerfile"))
# def t1() -> str:
#     return "hello"
```
