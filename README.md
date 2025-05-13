# Python Project Template

This repository provides a structured starting point for Python projects. It is designed to automate and standardize the setup process across different projects. The template includes a recommended folder structure, configuration files, and project automation through a `Makefile`.

## Steps

A local script on the user's machine (not included in this repository) automates the project initialization process. When run with a project name as input, it performs the following steps:

- Clones this template repository into a predefined parent directory under the specified project name
- Removes the existing `.git` history from the cloned folder
- Initializes a new Git repository
- Opens the newly created project folder in the default code editor (e.g. VSCode)
- Executes `make setup` to prepare the environment (create a virtual environment, install dependencies using `uv`, etc.)

This allows you to start working in a clean, fully configured project space within seconds.