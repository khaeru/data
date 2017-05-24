# Codes for research data sets

This repository contains Python 3 modules that retrieve, clean, subset and otherwise transform various data sets used in research. The objective is to abstract these tasks and keep them separate from research code that performs actual analysis on the data.

* `ceic` — CEIC Data's China Premium Database
* `chip` — China Household Income Project
* `chfs` — China Household Finance Survey
* `cn_nbs` — National Bureau of Statistics of China
* `naturalearth` — [Natural Earth](http://naturalearthdata.com) GIS

The modules are independent but have a roughly similar [API](https://en.wikipedia.org/wiki/Application_programming_interface). Each module…
- contains a method like `load_ceic()` that returns data in a clean, Pythonic form.
- may contain a method like `import_ceic()` that processes raw data sets into a cache in the directory of the name (e.g. `ceic/` for `ceic.py`).
- can be invoked as a command-line program, e.g. `python3 -m ceic`. Invoking a module without any arguments gives basic usage instructions, but the code is also documented.
- may make use of a configuration file in the directory of the same name. Example configuration files are provided.
