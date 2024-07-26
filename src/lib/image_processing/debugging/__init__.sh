#!/bin/bash

dn="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

cd "$dn"

# Create or clear the __init__.py file
echo "# Auto-generated __init__.py" > __init__.py

# Loop through all .py files in the current directory (non-recursive)
for pyfile in *.py; do
    # Skip the __init__.py itself
    [[ "$pyfile" == "__init__.py" ]] && continue

    # Get the module name (file name without extension)
    module="${pyfile%.py}"

    # Append the import statement to the __init__.py file
    echo "from .${module} import *" >> __init__.py
done

echo "Generated __init__.py"

