

# Recreate error

## Create dev environment
```
conda create --name flyte-eval-combo python=3.11 -c conda-forge
conda activate flyte-eval-combo
pip install ./flyte_science
```

## Start local flyte cluster
```
flytectl demo start
```

## Build docker images
```
for x in {1..3}; do docker build . -f Dockerfile.foo$x -t flyte-foo$x; done
```

## Submit wf & get error
```
pyflyte run flyte_science/scripts/one_off_combo_script.py main
```

see error at `error.txt`
