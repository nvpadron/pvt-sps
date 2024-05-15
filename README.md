# PVT-SPS
GNSS Standalone PVT Standard Positioning Service (GPS L1)

See detailed in description and usage in [PVT-SPS Documentation](https://www.spacewayfinder.com/docs/pvt-sps/index.html)


Nicolás Padrón - https://www.spacewayfinder.com/

This project is programmed in Python (/src).

## Table of Contents

- [Installation](#installation)
- [Clone repository](#clone-repository)
- [Python Tools](#python-tools)
- [Documentation](#documentation)
- [Usage example](#usage-example)
- [License](#license)

## Installation
### Clone repository
```bash
git clone https://github.com/nvpadron/pvt-sps.git
```
### Python Tools

Make sure to PIP install the needed libraries for using the python scripts in /tools folder. You can install the "requirements_venv.txt" after creating a virtual environment:

Crete environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```
Update PIP (might be required, if so, I recommend to do it before installing the python libraries):
```bash
python -m pip install --upgrade pip
```
Install the libraries:
```bash
pip install -r .\requirements.txt
```

## Documentation
Docuemtation can be found in https://www.spacewayfinder.com/docs/pvt-sps/index.html

## Usage example
Usage information can be found in section **Usage Description** on Documentation URL.
In short, explore the modes and configuration in /src/config.py file and run the code (from root directory) as
```bash
python .\src\main.py
```

## License
License file can be found in LICENSE.

------------------------