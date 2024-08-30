import importlib
import os
from typing import List, Union


def from_disconnected_import(
    containing_folder: str,
    prefix: Union[str, None],
    namechain: List[str],
    symbol_name: str,
):
    # Step 1: Determine the prefix
    if prefix is None:
        # Infer the prefix from the basename of the containing folder
        inferred_prefix = os.path.basename(os.path.realpath(containing_folder))
        # Validate the inferred prefix
        if not inferred_prefix.isidentifier():
            raise ValueError(
                f"Inferred prefix '{inferred_prefix}' is not a valid Python identifier."
            )
        prefix = inferred_prefix

    # Step 2: Construct the module name
    module_name = ".".join([prefix] + namechain)

    # Step 3: Construct the full path to the module file
    module_path = os.path.join(containing_folder, *namechain) + ".py"

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module file not found at '{module_path}'")

    # Step 4: Load the module from the file
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(
            f"Could not create a module spec for '{module_name}' from path '{module_path}'"
        )

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(
            f"Failed to load module '{module_name}' from '{module_path}'"
        ) from e

    # Step 5: Retrieve the symbol
    try:
        symbol = getattr(module, symbol_name)
        return symbol
    except AttributeError as e:
        raise ImportError(
            f"Module '{module_name}' does not have a symbol named '{symbol_name}'"
        ) from e
