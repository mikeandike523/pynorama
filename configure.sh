#!/bin/bash

dn="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"


cd "$dn"

python3 -m venv pyenv

source pyenv/bin/activate
python3 -m pip install pipenv
deactivate

chmod +x ./__pipenv
chmod +x ./__python
chmod +x ./pynorama