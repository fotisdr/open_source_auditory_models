## Python demonstration for VCCA2024

Simulate hearing loss on an audio signal and visualise the neural representation of sound (neural activity) in the brain using a deep neural network (DNN) auditory model. 

### Installation requirements

The first part of the demonstration (hearing loss simulation) requires the files of the [Clarity Challenge toolkit](https://github.com/claritychallenge/clarity) to run the MSBG model. The second part (central auditory processing simulation) requires an active installation of Tensorflow to execute the DNN model ([ICNet](https://github.com/fotisdr/ICNet)), as well as the files of the corresponding repository. Both Clarity and ICNet repositories are included as submodules of this repository. To execute the VCCA demo, we recommend installing conda from [here](https://www.anaconda.com/download) and then running the following commands on the anaconda command prompt to load the pre-compiled environment `vcca_demo` (with Python v3.11 and all the necessary packages):

```
conda env create -f vcca_demo.yml
conda activate vcca_demo
```

If loading the conda environment from the `vcca_demo.yml` file does not work, the following commands can be used to manually create the `vcca_demo` environment and install all packages. If not using conda, only the last two commands are needed to install all necessary packages on Python. 

```
conda create --name vcca_demo python=3.11
conda activate vcca_demo
pip install -r ICNet/requirements.txt
pip install omegaconf soundfile
```

### Example code

The `VCCA_demo.ipynb` notebook includes the example code for the Python demonstration and can be executed by running `jupyter lab` on the command prompt and navigating to the notebook (or using Jupyter notebook). It includes all the steps to read an example sound file (found under `ICNet/scribe_male_talker.wav`), simulate hearing loss using the MSBG model, listen to the generated sound file, and use the normal-hearing and hearing-loss sound files as inputs to simulate neural activity in the brain using ICNet. A pre-compiled Python version of the provided Jupyter notebook is also included (`VCCA_demo.py`), and an HTML file that includes all the generated plots from the Jupyter notebook (`VCCA_demo.html`).

### References

- **MSBG model**: Nejime, Y., & Moore, B. C. (1997). Simulation of the effect of threshold elevation and loudness recruitment combined with reduced frequency selectivity on the intelligibility of speech in noise. The Journal of the Acoustical Society of America, 102(1), 603–615. [https://doi.org/10.1121/1.419733](https://doi.org/10.1121/1.419733)

- **1st Clarity Enhancement Challenge**: Graetzer, S. N. et al. (2021). Clarity-2021 challenges: Machine learning challenges for advancing hearing aid processing. In Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH (Vol. 2). [https://doi.org/10.21437/Interspeech.2021-1574](https://doi.org/10.21437/Interspeech.2021-1574)

- **ICNet model**: Drakopoulos, F. et al. (2024). Modeling neural coding in the auditory brain with high resolution and accuracy. bioRxiv, 2024.06.17.599294. [https://doi.org/10.1101/2024.06.17.599294](https://doi.org/10.1101/2024.06.17.599294)
