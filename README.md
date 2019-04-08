# scAlign
A deep learning-based tool for alignment and integration of single cell genomic data across multiple datasets, species, conditions, batches

## Tutorials

[Unsupervised aligment and projection of HSCs](https://github.com/quon-titative-biology/examples/blob/master/scAlign_kowalcyzk_et_al/scAlign_kowalcyzk_et_al.md)

## Updates

DONE: Finalize partial labels code. 

Updated and validated (4/8/19)

-- To use partial labels: set unlabeled cells to be -1 in the label vector passed to scAlign. Specially, the scAlign.labels slot of the combined SCE object.

TODO: Supervised and partially supervised tutorial.

## Package requirements

Guide to installing python and tensorflow on different operating systems.

### On Windows:
1. Download Python 3.6.8. Note, newer versions of Python (e.g. 3.7) cannot use TensorFlow at this time. 
2. Make sure pip is included in the installation.
3. Open Windows Command Prompt, or PowerShell.
4. Navigate to your Python installation directory (for Windows 10, the default seems to be C:\Users\userid\AppData\Local\Programs\Python\Python36\Scripts, where userid is your own username).
5. Run .\pip install --upgrade tensorflow

### On Ubuntu:
1. sudo apt update
2. sudo apt install python3-dev python3-pip
3. pip3 install --user --upgrade tensorflow  # install in $HOME
4. python3 -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"

### On MacOS (homebrew):
1. /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
2. export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
3. brew update
4. brew install python  # Python 3
5. pip3 install --user --upgrade tensorflow  # install in $HOME
6. python3 -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"

Further details at: https://www.tensorflow.org/install
