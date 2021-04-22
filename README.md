# Evaluation of methods and tools for taxonomic NER (species name recognition)

This repository contains evaluation scripts, docker images and links to the corpora used for the paper **TaxoNERD: deep neural models for the recognition
of taxonomic entities in the ecological and evolutionary literature**.

## Corpora

The corpora can be publicly accessed at the following links:

| Corpora | Text Genre | Standard | Entities | Publication |
| --- | --- | --- | --- | --- |
| [Linnaeus](http://linnaeus.sourceforge.net/)| Scientific Article | Gold | species | [link](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-85)|
| [S800](http://species.jensenlab.org/)| Scientific Article | Gold | species|[link](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0065390)|
| [COPIOUS](http://www.nactem.ac.uk/copious/) | Scientific Article | Gold | taxon, geographical location, habitat, temporal expression, and person | [link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6351503/pdf/bdj-07-e29626.pdf)|
| [Bacteria Biotope](https://sites.google.com/view/bb-2019/task-description?authuser=0) | Scientific Article | Gold | microorganism, habitat, geographical location, phenotype | [link](https://www.aclweb.org/anthology/D19-5719.pdf)|

### Preprocessing

Corpora pre-processing operations were collected in a single jupyter notebook for ease-of-use.

### Train/test/dev split

- LINNAEUS: we used the train, test and validation sets of [Giorgi and Bader, 2018](https://github.com/BaderLab/Transfer-Learning-BNER-Bioinformatics-2018).
- S800: we used [this script](https://github.com/spyysalo/s800) to generate the subsets.
- COPIOUS: the COPIOUS corpus is already splitted into train, test and validation sets. 
- BB task: we used the validation set for testing, and randomly split the train set into train/validation subsets with a 85:15 ratio.

## Images

To facilitate the install of existing taxonomic NER tools written in different languages, we provide a Dockerfile for each tool. This means you will need [Docker](https://www.docker.com/) to run the evaluation scripts. Code for building Docker images is provided as part of the evaluation scripts, so you do not have to build the images yourself.

## Evaluation

All scripts used for evaluation are provided as jupyter notebooks, one per evaluated method.
