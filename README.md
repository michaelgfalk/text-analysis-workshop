# Text Analysis Workshop

An introductory workshop for students with some knowledge of Python and pandas.

## Installation

The best way to run the workshop is with `uv` and a virtual environment.

```{bash}
uv venv --python 3.15
source .venv/bin/activate
uv pip install -r requirements.txt
jupyter lab .
```

## Dependency Management

The workshop is designed for use on the ARDC BinderHub service. That service relies on conda for dependency management, while I use `uv`. Accordingly, there are `requirements.txt` and `.python-version` files as well as an `environment.yml` file. Note to self: don't allow these to get out of sync! (Unlikely...)

