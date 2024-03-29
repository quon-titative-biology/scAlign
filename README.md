# scAlign
A deep learning-based tool for alignment and integration of single cell genomic data across multiple datasets, species, conditions and batches

## Tutorials

First follow the install instructions below, at the bottom of the page, before following the tutorials.

[Unsupervised alignment and projection of HSCs](https://github.com/quon-titative-biology/examples/blob/master/scAlign_paired_alignment/scAlign_kowalcyzk_et_al.md)

[Multiway alignment using all pairs method](https://github.com/quon-titative-biology/examples/blob/master/scAlign_multiway_alignment/scAlign_multiway_pancreas.md)

[Supervised/Semi-supervised alignment](https://github.com/quon-titative-biology/examples/blob/master/scAlign_supervised_alignment/scAlign_supervised_alignment.md)

## Contributors

[Chang Kim](https://github.com/cnk113) | 
------------ |
<img src="https://avatars1.githubusercontent.com/u/21249710?v=4&s=25" width="150" height="150" /> | 


## Updates

#### (11/15/2021) Updated to Tensorflow 2. Now req. (Tensorflow >= 2.0)

#### (9/4/2019) Updated install instructions to include Tensorflow for R method.

#### (5/9/2019) Updated to version 1.0! Tutorials for multiple modes of operation now available. 

## R Package and Bioconductor

Bioconductor for now will only support the Linux version of scAlign. 

The latest version of scAlign for all systems can always be found at [github](https://github.com/quon-titative-biology/). 

```
install.packages('devtools')
devtools::install_github(repo = 'quon-titative-biology/scAlign')
library(scAlign)
```

## Package requirements

scAlign has three dependencies: Python 3, tensorflow (the R package), and tensorflow (the Python package). This is a guide to installing python and Tensorflow on different operating systems. 

### (Python)
  #### All platforms:
  1. [Download install binaries for Python 3 here](https://www.python.org/downloads/release/)
  #### Alternative (On Windows):
  1. Download Python 3
  2. Make sure pip is included in the installation.

  #### Alternative (On Ubuntu):
  1. sudo apt update
  2. sudo apt install python3-dev python3-pip

  #### Alternative (On MacOS):
  1. /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  2. export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
  3. brew update
  4. brew install python  # Python 3
  
### (Tensorflow)
In an R session:
  ```
  install.packages('tensorflow') #install the tensorflow R package (that sits on top of the TensorFlow python package)
  library(tensorflow)
  ```
