# DDFT-Spherical
DDFT-spherical is an implementation of dynamic density functional theory in the dynamic test particle limit for hard spheres.

## Getting started
We guide you through running an example simulation first.
There are two recommended options to run this project.

### a) Virtual Environment
We have a pipfile to specify all python dependencies. You need to install Python 3.9 and [Pipenv](https://pypi.org/project/pipenv/).
Then, run `pipenv install` to install all dependencies automatically into a newly created virtual environment.
This may take a while.
Once the installation is finished, run `pipenv shell` to enter a shell inside the virtual environment.
From there, starting `./run.sh` should run an example simulation.

### b) Docker
To ensure a known-working environment, you can also run this software via [Docker](https://www.docker.com/).
After you have installed Docker and [docker-compose](https://docs.docker.com/compose/),
 - Create a directory `docker` with subdirectories `config`, `out`, and `cache`.
   These directories will be mounted inside the docker image to have persistent storage of simulation results,
   configuration, and cache. If you want to use different paths, change the left-hand side of the `volumes`
   section of the `docker-compose.yml` configuration file.
 - Inside `docker/config`, place the file `config.jsonc` from the main directory.
 - Now, run `docker-compose up --build`. This will build a docker image and start it.
   Be prepared to wait awhile on the first build.
   On subsequent builds, the heavy part of this process is read from cache, so it should be much faster.

### Simulation
No matter which option you chose above, you should now have a simulation running.
You should expect the simulation itself to finish within a few hours, but you can check the intermediate progress.
In the directory `out` or `docker/out` (depending on your setup), you will find a file `diffusion.dat` which contains
a series of density profiles, separated by double newlines.
You can plot these e.g. with [gnuplot](http://www.gnuplot.info/).
Congratulations, you just did you first DDFT calculation!

## Configuration
The simulation is configured via the configuration file `config.jsonc`.
A different filename can be chosen by setting the environment variable `$DDFT_CONFIG` in `docker-compose.yml` if using
docker, or setting this variable in your shell, if using the virtual environment.
Please refer to the provided example configuration file for an explanation of the available configuration options.

## Initial Density Profiles
The provided density profile represents hard spheres at bulk density \rho = 1090/(1500\sigma^3) \approx 0.73\sigma^{-3}
and packing fraction \eta \approx 0.38. You can provide your own initial density profile as a compressed numpy file.
`initial.py` has a function `export` which helps you create a file in the correct format.

## Extending and Adapting
You can build upon this project and e.g. implement your own excess free energy functional.
To do this, you need these steps:
 - create a new file inside the folder `fexc`.
 - in that file, create a new class that inherits from the class `Fexc`:
   ```
   from fexc import Fexc
   
   class MyFancyFunctional(Fexc):
       pass
   ```
   You need to implement at least `d_fexc_d_rho(self, rho: (np.array, np.array)) -> (np.array, np.array)`,
   which takes as arguments a tuple of numpy arrays (where the first is the self density profile and the
   second is the distinct density profile), and returns the functional derivatives of your functional with
   respect to each of these density profiles.
   You may also provide the functions `fexc(self, rho: (np.array, np.array)) -> float`, which should return
   the value of your functional evaluated at the given density profiles, and
   `fexc_and_d_fexc(self, rho: (np.array, np.array)) -> (float, np.array, np.array)`,
   which calculates both the value of the functional and the value of the functional derivatives.
   The latter two functions are currently unused by the integration algorithm, but may be helpful when
   implementing unit tests.
 - finally, extend the loader function in `fexc/loader.py`. Think of a name for your functional and extend
   the if-elif-clause to load your functional. You can then specify its name in the configuration file.

Happy hacking! Feel free to raise a pull request to share your awesome new functional.

## Testing
We have an extensive test suite for the basic functionality of this software.
To run the test suite, set up a virtualenv as specified in the "Getting started" section above.
In principle, it is also possible to run the tests via Docker, but if you are interested in the 
tests, you are likely going to want to do your own development, which is cumbersome if using Docker.
Enter the virtualenv using `pipenv shell`, then run `python run_pytest.py`.
The tests will take some time to complete.

If you implement your own functional, we encourage you to also write some tests for it.
It may seem cumbersome at first, but will help you in the long run.
If you modify any of the other functionality of this software, run the tests to find any breakage you may have caused.
If this is the case, likely you have created a bug.
The test suite is there to help you find it.
If the breakage is not a bug, but some intended change of behaviour, please adapt the affected unit tests as well!
This also helps others to understand your intentions.
