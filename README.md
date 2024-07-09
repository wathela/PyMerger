[![Language](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/wathela/pymerger/blob/main/LICENSE)

# PyMerger

PyMerger is a Python tool for detecting binary black hole mergers from the Einstein
Telescope, based on a Deep Residual Neural Network model.

## Overview

PyMerger is a Python tool that uses a Deep Residual Neural Network model to identify BBH mergers without the need for data whitening or band-passing. 
The model is trained on three sub-detectors combined data of ET.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/PyMerger.git
   cd PyMerger

2. Install the [required Python packages](requirements.txt):
   ```sh
   pip install -r requirements.txt 
## Usage

```sh
usage: pymerger.py [-h] [-r {8192,4096}] [-n NO_SEGMENT] [-c CHANNELS CHANNELS CHANNELS] [-t THRESHOLD] -i INPUT_FILE_DIR -f OUTPUT_DIR [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  -r {8192,4096}, --sampling-rate {8192,4096}
                        Sampling rate of the input data (either 8192 or 4096). Default is 8192.
  -n NO_SEGMENT, --no-segment NO_SEGMENT
                        Number of data segments to be processed for each detector (i.e., number of .gwf files to be processed for each detector).
                        Files in the input directory will be sorted, and the first 'n' files up to the specified number of segments will be processed.
                        Default is 1 which means there are 1 unique file from each detector.
  -c CHANNELS CHANNELS CHANNELS, --channels CHANNELS CHANNELS CHANNELS
                        List of the THREE channels to be processed. Default is ['E1:STRAIN', 'E2:STRAIN', 'E3:STRAIN'].
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold value for merger detection. A value between 0.1 and 0.5, where a smaller value will result in fewer detections but
                        a lower false positive rate. Default is 0.4.
  -i INPUT_FILE_DIR, --input-file-dir INPUT_FILE_DIR
                        Directory containing the input .gwf files.
  -f OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to store the results.
  --verbose             Enable verbose mode to print update messages. Default is true.
```
### Example: 
```sh
   python pymerger.py -r 8192 -n 2 -c E1:STRAIN E2:STRAIN E3:STRAIN -t 0.5 -i /path/to/input/files -f /path/to/output/dir
```

### Contact
Email: wathelahamed@gmail.com
