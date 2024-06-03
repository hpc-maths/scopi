SCoPI
=====

Installation
------------

- Install Mambaforge (https://github.com/conda-forge/miniforge#mambaforge)

  anaconda can be used if it's already installed on your system.

- Clone this repository

```
git clone https://gitlab.labos.polytechnique.fr/scopi/scopi.git
```

- Go into this directory

```
cd scopi
```

- Install the dependencies of **SCoPI** using a conda environment

```
mamba env create -f conda/environment.yml
```
or
```
conda env create -f conda/environment.yml
```

- Activate the environment

```
mamba activate scopi-env
```
or
```
conda activate scopi-env
```

- Configure **SCoPI** using CMake

```
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DSCOPI_USE_MKL=ON
```

- build **SCoPI**

```
cmake --build build
```

- Install **SCoPI** in the conda environment

```
cmake --install build
```

Use **SCoPI** in an external project
--------------------------------

You can use CMake to build your project with a dependency to **SCoPI**.
Here is just a quick example with one cpp file which includes **SCoPI** tools.

```cmake
find_package(scopi)

add_executable(test_scopi test_scopi.cpp)
target_link_libraries(test_scopi scopi)
```
