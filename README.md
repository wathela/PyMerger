[![Language](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/wathela/pymerger/blob/main/LICENSE)

# PyMerger

PyMerger is a Python tool for detecting binary black hole mergers from the Einstein
Telescope, based on a Deep Residual Neural Network model.

## Overview

We present PyMerger, a Python tool for detecting binary black hole (BBH) mergers from the Einstein Telescope (ET), based on a Deep Residual Neural Network model (ResNet). ResNet was trained on data combined from all three proposed sub-detectors of ET (TSDCD) to detect BBH mergers. Five different lower frequency cutoffs (F_low): 5 Hz, 10 Hz, 15 Hz, 20 Hz, and 30 Hz, with match-filter Signal-to-Noise Ratio (MSNR) ranges: 4-5, 5-6, 6-7, 7-8, and >8, were employed in the data simulation. Compared to previous work that utilized data from single sub-detector data (SSDD), the detection accuracy from TSDCD has shown substantial improvements, increasing from 60%, 60.5%, 84.5%, 94.5% to 78.5%, 84%, 99.5%, 100%, and 100% for sources with MSNR of 4-5, 5-6, 6-7, 7-8, and >8, respectively. The ResNet model was evaluated on the first Einstein Telescope mock Data Challenge (ET-MDC1) dataset, where the model demonstrated strong performance in detecting BBH mergers, identifying 5,566 out of 6,578 BBH events, with optimal SNR starting from 1.2, and a minimum and maximum D_L of 0.5 Gpc and 148.95 Gpc, respectively. Despite being trained only on BBH mergers without overlapping sources, the model achieved high BBH detection rates. Notably, even though the model was not trained on BNS and BHNS mergers, it successfully detected 11,477 BNS and 323 BHNS mergers in ET-MDC1, with optimal SNR starting from 0.2 and 1, respectively, indicating its potential for broader applicability.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/PyMerger.git
   cd PyMerger

2. Install the [required Python packages](requirements.txt):
   ```sh
   pip install -r requirements.txt 
## Usage
PyMerger assumes that each sub-detector of ET will have a separate .gwf file in three separate directories (E1, E2, E3). 
The data input path should point to the folder where these three directories are located.

```sh
usage: pymerger.py [-h] [-r {8192,4096}] [-n NO_SEGMENT] [-c CHANNELS CHANNELS CHANNELS] [-t THRESHOLD] -i INPUT_FILE_DIR -f OUTPUT_DIR [--verbose]

optional arguments:
  '-h, --help            show this help message and exit
  -r {8192,4096}, --sampling-rate {8192,4096}
                        Sampling rate of the input data (either 8192 or 4096). Default is 8192.
  -n NO_SEGMENT, --no-segment NO_SEGMENT
                        Number of data segments to be processed for each detector (i.e., number of .gwf files to be processed for each detector).
                        Files in the input directory will be sorted, and the first 'n' files up to the specified number of segments will be processed.
                        Default is 1 which means there are 1 unique file from each detector.
  -c CHANNELS CHANNELS CHANNELS, --channels CHANNELS CHANNELS CHANNELS
                        List of the THREE channels to be processed. Default is ['E1:STRAIN', 'E2:STRAIN', 'E3:STRAIN'].
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold value for merger detection. A value between 0.4 and 1, where a smaller value will result in fewer detections but
                        a lower false positive rate. Default is 0.6.
  -i INPUT_FILE_DIR, --input-file-dir INPUT_FILE_DIR
                        Directory containing the input .gwf files.
  -f OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to store the results.
  --verbose             Enable verbose mode to print update messages. Default is true.'
```
### Example: 
```sh
   python pymerger.py -r 8192 -n 2 -c E1:STRAIN E2:STRAIN E3:STRAIN -t 0.5 -i /path/to/input/files -f /path/to/output/dir
```

### Contact
Email: wathelahamed@gmail.com
