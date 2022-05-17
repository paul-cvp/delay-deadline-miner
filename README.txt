Steps for running DisCoveR-py within the docker container:
    - Open the CLI (Command Line Interface) of the Docker Container discover-py
    - Run "python main.py -f -a" to replicate the experiments of the paper (depending on your machine it might be an overnight job)
    - In powershell/bash do 'docker ps' and copy the 'CONTAINER ID' of the container with the name "discover-py"
    - In poweshell/bash do 'docker cp {CONTAINER ID}:/DisCoveR-py/models .' to copy all the results from the containers models folder in the current host folder (replace the '.'(dot) in the command to use a specific folder)
        -The results are suffixed as follows:
            - "{dataset name}_model" files contain the DCR graphs
            - "{dataset name}_timings" folders contain image files ".jpg" of the timing data:
                - "{CONDITION|RESPONSE}_{event from}_{event to}_hist" for histograms
                - "{CONDITION|RESPONSE}_{event from}_{event to}_boxplot" for boxplots
                - "{CONDITION|RESPONSE}_{event from}_{event to}_simple_fit" for the best 5 single parametric distribution fit from the Fitter library
                - "{CONDITION|RESPONSE}_{event from}_{event to}_advanced_fit" (only applicable for the subset of the road traffic fine dataset mined conditions and responses that have advanced fitting initial parameters specified in the advanced_timings_fit method of the main.py file)



Experiment setup:

    The code was run on Linux Ubuntu 20.4 OS inside a Windows 10 Subsystem for Linux WSL2 installation with the following specs:
    Processor: Intel(R) Core(TM) i7-7660U CPU @ 2.50GHz   2.50 GHz
    RAM: 16GB
    64-bit OS, x64-based processor
    Intel integrated graphics

    The expected runtime is around 8 hours.

Manual Steps (not necessary unless something goes wrong):

Prerequisites:

    Make sure that the project folder contains the following directory structure:
    DisCoveR-py
        data
        discover
        models
        main.py

Using Docker run the Dockerfile

- (done in the Dockerfile) Place the event log files (.xes) downloaded and unarchived from their ".gz" format from the following links inside the "data" folder:
    - https://data.4tu.nl/ndownloader/files/24073733
    - https://data.4tu.nl/ndownloader/files/24027287
    - https://data.4tu.nl/ndownloader/files/24063818
    - https://data.4tu.nl/ndownloader/files/24018146

- Run main.py with arguments:
    - '-f' or '--fine' for creating the advanced timing distributions fit from the road traffic fine dataset
    - '-a' or '--all' for creating the summary statistics for all the 4 event logs

- Make sure the Docker container is running. In powershell/bash do
    - 'docker ps' and copy the 'CONTAINER ID' of the container with the name "discover-py"
    - 'docker cp {CONTAINER ID}:/DisCoveR-py/models .' to retrieve all the results from the models folder in the current folder (replace the '.'(dot) in the command to use a specific folder)