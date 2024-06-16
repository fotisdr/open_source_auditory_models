## Open-source auditory models of normal-hearing and hearing-impaired processing

**Fotios Drakopoulos, Alejandro Osses**

20 June 2024, Virtual Conference on Computational Audiology (VCCA2024)

### Auditory model demonstrations

As part of VCCA 2024, we demonstrate how existing computational models and recent machine-learning approaches can be used to simulate auditory processing for normal-hearing and hearing-impaired listeners. We provide two demos of openly accessible auditory models on Python and MATLAB, respectively. 
- **MATLAB demo**: This demonstration shows how a monaural auditory model can be used with the AMT toolbox to simulate sound processing in the auditory system, comparing the effects of normal-hearing processing and those of a hearing impaired cochlea. Instructions, results and all necessary files to run the MATLAB demo are included under the `MATLAB` folder.
- **Python demo**: This demonstration shows how hearing loss can be simulated on an audio signal and how deep learning can be used to simulate the neural representation of sound in the brain. Instructions, results and all necessary files to run the Python demo are included under the `Python` folder.

### List of available auditory models

An overview of widely used open-source auditory models can be found in [Osses A et al. (2022). A comparative study of eight human auditory models of monaural processing.
Acta Acustica](https://acta-acustica.edpsciences.org/articles/aacus/full_html/2022/01/aacus210060/aacus210060.html), which compares eight models from the Auditory Modelling Toolbox (AMT). A more extensive list is given below and includes most auditory models that the authors have known and used. Note that many more auditory models exist which are not mentioned here. 

Each of the auditory models listed below has been developed for different purposes and might thus be better suited for different applications. The included models can be roughly grouped into four categories:
- **Cochlear filterbanks**: Time-frequency representations of sound that are fast to execute and can easily be used as front-ends in audio applications where real-time processing is needed. Examples include the MFCC and the Gammatone filterbank.
- **Functional models**: Computationally efficient non-linear models that target the simulation of perceptual outcomes (e.g. speech intelligibility) rather than the direct simulation of neural representations in the auditory system. Examples include the models by Dau et al. (1997) and Relaño-Iborra et al. (2019).
- **Biophysically inspired models**: Complex models that aim to simulate auditory processing by describing the physiological properties of the auditory system. Examples include the models by Verhulst et al. (2018), Zilany et al. (2014) and Bruce et al. (2018). Note that the definition of biophysically inspired models adopted here is more general, grouping into the same model family the phenomenological and the biophysical models according to Osses et al. (2022).
- **Deep learning models**: Models that are developed to learn the non-linear mapping of sound to neural activity directly from data. See ICNet. 

| Name  | Developers | Publication | Programming language | Notes |
| --- | --- | --- | --- | --- |
|[Auditory Modelling Toolbox (AMT)](https://www.amtoolbox.org/) | Various, coordinated by Majdak P | Various, toolbox [here](https://doi.org/10.1051/aacus/2022011), selection of monaural models [here](https://doi.org/10.1051/aacus/2022008) | MATLAB (Python / C++)| A MATLAB interface for using a variety of auditory models for normal-hearing and hearing-impaired auditory processing (including some of the models listed below). | 
|[Auditory Toolbox](https://engineering.purdue.edu/~malcolm/interval/1998-010/) | Slaney M et al. | Various | MATLAB / Python | Implementations of various auditory time-frequency representations such as Gammatone and MFCC filterbanks, the Ray Meddis model and the CARFAC model. A Python implementation is also available [here](https://github.com/MalcolmSlaney/python_auditory_toolbox), with support for PyTorch and JAX. 
|[Auditory models from Carney Lab](https://www.urmc.rochester.edu/labs/carney/publications-code/auditory-models.aspx) | Carney L et al. | Various, main publications: [here](https://doi.org/10.1121/1.4837815) and [here](https://doi.org/10.1523/ENEURO.0004-15.2015) | MATLAB | Phenomenological models of the auditory periphery, including the Zilany et al. (2014) model which can simulate inner-hair-cell and auditory-nerve responses to a sound with various degrees of hearing loss. | 
|[Auditory model from Bruce Lab](https://www.ece.mcmaster.ca/~ibruce/zbcANmodel/zbcANmodel.htm) | Bruce IC et al. | Bruce IC et al. (2018). *A phenomenological model of the synapse between the inner hair cell and auditory nerve: Implications of limited neurotransmitter release sites*, Hearing Research. | MATLAB | Simulates inner-hair-cell and auditory-nerve responses to a sound, and can include loss of outer hair cells, inner hair cells or auditory synapses. | 
|[Auditory model from the Hearing Technology Lab](https://github.com/HearingTechnology/Verhulstetal2018Model) | Verhulst S et al. | Verhulst S et al. (2018). *Computational modeling of the human auditory periphery: Auditory-nerve responses, evoked potentials and hearing loss*. Hearing Research. | MATLAB / Python | Simulates cochlear, inner-hair-cell, auditory-nerve and brainstem responses to a sound, including loss of outer hair cells or auditory synapses. A faster and differentiable implementation of the model based on deep learning can be found [here](https://github.com/HearingTechnology/CoNNear_periphery).| 
|[Brian hears auditory modelling library](https://github.com/brian-team/brian2hears) | Goodman DF et al. | Fontaine B et al. (2011). *Brian hears: online auditory processing using vectorization over channels*. Front Neuroinform. | Python | Includes several linear and non-linear models of the middle ear and the cochlea.| 
|[NSL Auditory-Cortical MATLAB Toolbox](http://nsl.isr.umd.edu/downloads.html) | Shamma S et al. | Chi T et al. (2005). *Multiresolution spectrotemporal analysis of complex sounds*. JASA. | MATLAB | Simulates auditory processing at different stages of the auditory pathway, from the early auditory pathway up to the brain. | 
| [Cambridge hearing loss simulator (MSBG)](https://github.com/claritychallenge/clarity/tree/main/clarity/evaluator/msbg) | Moore B, Stone M, Baer T, Glasberg B | Nejime Y et al. (1997). *Simulation of the effect of threshold elevation and loudness recruitment combined with reduced frequency selectivity on the intelligibility of speech in noise*. JASA. | MATLAB / Python | Simulates hearing loss on the audio waveform. The Python implementation from the Clarity Challenge is provided. |
| [ICNet](https://github.com/fotisdr/ICNet) | Drakopoulos F et al. | Drakopoulos F et al. (2024). *ICNet: a novel framework for modeling central auditory processing with high resolution and accuracy*. | Python / Tensorflow | A deep learning model that simulates normal-hearing neural activity in the inferior colliculus in response to a sound input. |


----

## Citation
If you use this code, please cite this repository:

Drakopoulos, F. & Osses, A. (2024). Open-source auditory models of NH and HI processing (v1.0). Zenodo. [https://zenodo.org/doi/10.5281/zenodo.11843926](https://zenodo.org/doi/10.5281/zenodo.11843926)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11843927.svg)](https://doi.org/10.5281/zenodo.11843927)

##
For questions, please reach out to one of the corresponding authors:

* Fotios Drakopoulos: f.drakopoulos@ucl.ac.uk
* Alejandro Osses: alejandro.osses@sorama.eu
