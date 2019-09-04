# scAlign
A deep learning-based tool for alignment and integration of single cell genomic data across multiple datasets, species, conditions, batches

## Tutorials

[Unsupervised alignment and projection of HSCs](https://github.com/quon-titative-biology/examples/blob/master/scAlign_paired_alignment/scAlign_kowalcyzk_et_al.md)

[Multiway alignment using all pairs method](https://github.com/quon-titative-biology/examples/blob/master/scAlign_multiway_alignment/scAlign_multiway_pancreas.md)

[Supervised/Semi-supervised alignment](https://github.com/quon-titative-biology/examples/blob/master/scAlign_supervised_alignment/scAlign_supervised_alignment.md)

## Updates

#### (9/4/2019) Updated install instructions to include Tensorflow for R method..

#### (5/9/2019) Updated to version 1.0! Tutorials for multiple modes of operation now available. 

## R Package and Bioconductor

Bioconductor for now will only support the Linux version of scAlign. 

The latest version of scAlign for all systems can always be found at [github](https://github.com/quon-titative-biology/). 

```
install.packages('devtools')
devtools::install_github(repo = 'quon-titative-biology/scAlign')
library(scAlign)
```
## Package requirements (Tensorflow for R)

Guide to installing tensorflow from R assuming a pre-existing version of Python on the system. 

```
library(tensorflow)
install_tensorflow(version = "gpu") ## Removing version will install CPU version of Tensorflow
```

## Package requirements

Guide to installing python and Tensorflow on different operating systems.

### Python:
  #### On Windows:
  1. Download Python 3.6.8. Note, newer versions of Python (e.g. 3.7) cannot use TensorFlow at this time. 
  2. Make sure pip is included in the installation.

  #### On Ubuntu:
  1. sudo apt update
  2. sudo apt install python3-dev python3-pip

  #### On MacOS (homebrew):
  1. /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  2. export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
  3. brew update
  4. brew install python  # Python 3
  
### Tensorflow:
  ```
  library(tensorflow)
  install_tensorflow(version = "gpu") ## Removing version will install CPU version of Tensorflow
  ```
