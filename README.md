# Mean-Field Control for Droplet Dynamics
This README file provides an overview of the repository and instructions on how to build and use the provided codes to compute the FEM/JKO schemes for the lubrication model (in NGSolve), and the mean-field control of droplet dynamics (in MFEM).
These codes accompany the manuscript [here](http://https://arxiv.org/abs/2402.05923). 

# MFEM code: PDHG Algorithm for MFC
To use the MFEM code in this repository, you need to install and set up the necessary dependencies, with the core requirement being the parallel version of the MFEM library. We will use the jkop-mfem branch of the MFEM fork available [here](https://github.com/pazner/mfem/tree/mfem-jkop). Building the parallel version of MFEM requires an MPI C++ compiler and external libraries such as hypre and METIS. Below are the instructions to download and install MFEM along with its prerequisites. A more detailed version can be found [here](https://mfem.org/building/#parallel-mpi-version-of-mfem).

  1. Create an empty directory 'MFEM/' and set as current working directory:
     ```
     mkdir MFEM
     cd MFEM
     ```
  2. Download and build hypre:
     ```
     git clone https://github.com/hypre-space/hypre.git 
     cd hypre/src
     ./configure && make install
     cd ../../
     ```
  3. Download and build METIS (4.0.3)
     ```
     wget https://github.com/mfem/tpls/raw/gh-pages/metis-4.0.3.tar.gz
     tar -zxvf metis-4.0.3.tar.gz 
     cd metis-4.0.3
     make OPTFLAGS=-Wno-error=implicit-function-declaration
     cd ../
     ln -s metis-4.0.3 metis-4.0
     ```
  4. Download and build the mfem-jkop branch
     ```
     git clone --single-branch --branch mfem-jkop https://github.com/pazner/mfem.git
     cd mfem
     make parallel -j 4 MFEM_USE_ZLIB=YES
     cd ../
     ```
Having successfully built all the dependencies, you can proceed to build the code in this repository by following these instructions:
 1. Clone the repository
    ```
    git clone https://github.com/gridfunction/DROPLET_MFC.git
    cd DROPLET_MFC
    ```
 2. Set the MFEM path in Make.user file:
    ```
    MFEM_DIR = ../mfem
    ```
    (Create the file if it does not exist and set MFEM_DIR)
 3. Compile the code
    ```
    make
    ```
# Usage
From within the `DROPLET_MFC` directory, you can run a mfc computation of different droplet actuations 
and the approx. JKO scheme with PDHG solver:
```
mpirun -np 16 -bind-to core:2 ./jko_lub -tC 1 
mpirun -np 16 -bind-to core:2 ./drop -tC 34 
```


# NGSOLVE code
drop_fem.py : FEM with BDF2 for the lubrication model
drop_jko.py : Approx. JKO scheme for the lubrication model

# Install a recent Python. Then it should be easy to install NGSolve using
```
>> pip install jupyter numpy scipy matplotlib
>> pip install --pre ngsolve
>> pip install webgui_jupyter_widgets
```
see more ngsolve documentation [here](https://ngsolve.org)

To run the code, type
```
python3 drop_fem.py 
```
or 
```
python3 drop_jko.py 
```

