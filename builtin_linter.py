from pkgutil import iter_modules
from sys import builtin_module_names

abc_module_finder = next(module_info for module_info in iter_modules() if module_info.name == "abc").module_finder
module_names = [*list(builtin_module_names), *[module_info.name for module_info in iter_modules() if module_info.module_finder == abc_module_finder]]

# https://docs.python.org/3/library/__main__.html#module-__main__ should be omitted from checking, are there other modules that should be as well?
# https://flake8.pycqa.org/en/latest/plugin-development/index.html this ultimately needs to become a flake8 plugin
# https://github.com/gforcada/flake8-builtins/blob/master/flake8_builtins.py might be able to just extend this one somehow, add these names to the list and add checking for filenames
# https://discuss.python.org/t/warning-when-importing-a-local-module-with-the-same-name-as-a-2nd-or-3rd-party-module this has relevant code examples
# should this also cover whatever 3rd party libraries are installed and/or in our pinned-requirements.txt files?
