# Text Analysis Workshop

An introductory workshop for students with some knowledge of Python and pandas.

## Run in Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/michaelgfalk/text-analysis-workshop/HEAD?urlpath=%2Fdoc%2Ftree%2Frelease%2Fworkshop%2Ftext-analysis-workshop.ipynb)

This is the easiest way to run the workshop. Open the student version of the workshop in Binder, will all dependencies installed.

## Installation

The best way to run the workshop locally is with `uv` and a virtual environment.

```{bash}
uv venv --python 3.15
source .venv/bin/activate
uv pip install -r requirements.txt
jupyter lab .
```

## Dependency Management

The workshop is designed for use on the ARDC BinderHub service. That service relies on conda for dependency management, while I use `uv`. Accordingly, there are `requirements.txt` and `.python-version` files as well as an `environment.yml` file. Note to self: don't allow these to get out of sync! (Unlikely...)

