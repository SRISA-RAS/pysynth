# pysynth

A prototype for the test program generator that creates assembly sources based on the execution trace of the real workload.
The result is supposed to resemble the characteristic features of the original program.
To be used in the verification process of microprocessor models.

## Running

Create a virtual environment and activate it:
```
python -m venv env
source ./env/bin/activate
```

Install the dependencies:
```
pip install -r requirements.txt
```

To be able to use graph image generation, install the `graphviz` dependency using the system package manager.
For example, for the Debian/Ubuntu one should do:
```
sudo apt-get install graphviz graphviz-dev
```

To see the available command line options:
```
python . --help
```

To generate the test in the directory `output` from the log `logs/log1a.txt`, launch the script with:
```
python . logs/log1a.txt -o output
```

To troubleshoot potential issues (e.g. the program is stuck somewhere, looping endlessly), one may increase verbosity:
```
python . logs/log1a.txt -o output -v
```

## Issues

At the moment only a small part of the program is working correctly. For example, one is required to run it with the argument `--loops none`, as otherwise it will fail:
```
python . logs/log1a.txt -o output --loops none
```
This specific problem comes from particular lack of success with our attempts to 'reduce' loops and somehow reproduce their behaviour from the original workload.
Firstly, a more elaborate analysis is required before making any changes on the loop. Secondly, it is hard to specify the algorithm for proper execution of those re-generated loops
that would always produce correct results.
