from ensurepip import version
import pkg_resources
import os
import argparse
import gc
import time
from tqdm import tqdm
from contextlib import redirect_stdout
import logging
logging.getLogger('pycbc').setLevel(logging.ERROR)
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from gwpy.timeseries import TimeSeries
import scipy as sp
from typing import List

def load_model(path: str) -> tf.lite.Interpreter:

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    logging.info("Model loaded.")
    return interpreter



class ETFileScanner:
    def __init__(self, data_path: str, 
                 num_seg: int, 
                 channels: List[str], 
                 threshold: float, 
                 window_size: int, 
                 xsize: int, 
                 detector: tf.lite.Interpreter, 
                 result_path: str, 
                 verbose: bool, 
                 batch_size: int = 50) -> None:
        
        self.data_path = data_path
        self.num_seg = num_seg
        self.channels = channels
        self.threshold = threshold
        self.window_size = window_size
        self.xsize = xsize
        self.fs = 16384
        self.detector = detector
        self.result_path = result_path
        self.batch_size = batch_size
        self.verbose = verbose

    def predict(self, data: np.ndarray) -> np.ndarray:
        input_data = np.array(data, dtype=np.float32) 
        input_details = self.detector.get_input_details()
        output_details = self.detector.get_output_details()
        self.detector.set_tensor(input_details[0]['index'], input_data)

        self.detector.invoke()

        return self.detector.get_tensor(output_details[0]['index'])
    
    def scale_minmax(self, X: np.ndarray, mn: float = 0.0, mx: float = 1.0) -> np.ndarray:
        return (X - X.min()) / (X.max() - X.min()) * (mx - mn) + mn

    def ts_to_img(self, ts: TimeSeries) -> np.ndarray:
        Sxx = sp.signal.spectrogram(x=np.array(ts), fs=self.fs, nfft=1024, mode='magnitude')[2]
        return np.flip(self.scale_minmax(Sxx), axis=0)[-42:, :]

    def get_slice(self, ET_data: TimeSeries, start_time: float, data_end: float, window_size: float) -> TimeSeries:
        if data_end >= window_size:
            ET_seg = ET_data.time_slice(start=start_time, end=window_size)
        else:
            ET_seg = ET_data.time_slice(start=abs(data_end - window_size), end=data_end)
        return ET_seg

    def save_results(self,  start_time: float, end_time: float, prob: float, cls: int, file_name: str) -> None:
        with open(os.path.join(self.result_path, file_name + '.txt'), 'a') as f:
            f.write("{:.3f},{:.3f},{:.4f},{}\n".format(float(start_time), float(end_time), prob, cls))

    def find_class(self, result: float) -> int:
        return 1 if result >= self.threshold else 0

    def get_sorted_files(self,  input_path: List[str]) -> List[str]:
        # directories
        sub_detectors = ['E1', 'E2', 'E3']
        sorted_files = {}

        for detector in sub_detectors:

            detector_path = os.path.join(input_path, detector)
            
            # Check if the directory exists
            if not os.path.isdir(detector_path):
                raise FileNotFoundError(f"Directory {detector_path} does not exist")
            
            files = [os.path.join(detector_path, f) for f in os.listdir(detector_path) if os.path.isfile(os.path.join(detector_path, f))]
            files.sort()
            sorted_files[detector] = files[:self.num_seg]
        
        return sorted_files

    def scan_files(self) -> None:
        data_files = self.get_sorted_files(self.data_path)
        for E1_file,E2_file, E3_file in tqdm(zip(data_files['E1'], data_files['E2'], data_files['E3']), total=len(data_files['E3'])):
        # for data_file in tqdm(data_files):
            file_suffix = os.path.basename(E1_file)[4:]
            if self.verbose:
                logging.info(f"Processing: {file_suffix} ...")

            try:
            
                # Load strain data for three detectors
                ET_data1 = TimeSeries.read(E1_file, self.channels[0]).to_pycbc()
                ET_data2 = TimeSeries.read(E2_file, self.channels[1]).to_pycbc()
                ET_data3 = TimeSeries.read(E3_file, self.channels[2]).to_pycbc()
                
                start_time = ET_data1.start_time
                data_end = ET_data1.end_time
                slide_window = start_time + self.window_size

                result_file_name = "result" + os.path.splitext(file_suffix)[0] + "_"
            
                for i in range(round(ET_data1.duration / self.window_size)):
                    E123_inj = np.zeros((42, self.xsize, 3))

                    ET1_slice = self.get_slice(ET_data1, start_time, data_end, slide_window)
                    ET2_slice = self.get_slice(ET_data2, start_time, data_end, slide_window)
                    ET3_slice = self.get_slice(ET_data3, start_time, data_end, slide_window)

                    slide_window += self.window_size
                    start_time += self.window_size

                    img1 = self.ts_to_img(ET1_slice)
                    img2 = self.ts_to_img(ET2_slice)
                    img3 = self.ts_to_img(ET3_slice)

                    E123_inj[:, :, 0] = img1
                    E123_inj[:, :, 1] = img2
                    E123_inj[:, :, 2] = img3

                    reshaped_image = np.expand_dims(E123_inj, axis=0)
                    prediction = self.predict(reshaped_image)
                    classification = self.find_class(prediction[0])

                    if classification == 0:
                        self.save_results(ET1_slice.start_time, ET1_slice.end_time, 1-prediction[0][0], 0, result_file_name)
                
                    # Clean up variables to free memory
                    del ET1_slice, ET2_slice, ET3_slice, img1, img2, img3, E123_inj
                    

                # Clean up main data variables to free memory
                del ET_data1, ET_data2, ET_data3
                gc.collect()
            except Exception as e:
                if self.verbose:
                    logging.error(f"Failed to process {file_suffix}: {str(e)}")
                with open('failed_files.txt', 'a') as f:
                    with redirect_stdout(f):
                        print(f"{file_suffix}: {e}")
                f.close()

                continue

    def process_batch(self, files_batch):
        for data_file in files_batch:
            file_suffix = data_file[4:]
            if self.verbose:
                logging.info(f"Processing: {file_suffix} ...")

            try:
                ET_data1 = TimeSeries.read(os.path.join(self.data_path, f'E1/E-E1{file_suffix}'), self.channels[0]).to_pycbc()
                ET_data2 = TimeSeries.read(os.path.join(self.data_path, f'E2/E-E2{file_suffix}'), self.channels[1]).to_pycbc()
                ET_data3 = TimeSeries.read(os.path.join(self.data_path, f'E3/E-E3{file_suffix}'), self.channels[2]).to_pycbc()

                start_time = ET_data1.start_time
                data_end = ET_data1.end_time
                slide_window = start_time + self.window_size

                result_file_name = "result" + os.path.splitext(file_suffix)[0] + "_"
            
                for _ in range(round(ET_data1.duration / self.window_size)):
                    E123_inj = np.zeros((42, 91, 3))

                    ET1_slice = self.get_slice(ET_data1, start_time, data_end, slide_window)
                    ET2_slice = self.get_slice(ET_data2, start_time, data_end, slide_window)
                    ET3_slice = self.get_slice(ET_data3, start_time, data_end, slide_window)

                    slide_window += self.window_size
                    start_time += self.window_size

                    img1 = self.ts_to_img(ET1_slice)
                    img2 = self.ts_to_img(ET2_slice)
                    img3 = self.ts_to_img(ET3_slice)

                    E123_inj[:, :, 0] = img1
                    E123_inj[:, :, 1] = img2
                    E123_inj[:, :, 2] = img3

                    reshaped_image = np.expand_dims(E123_inj, axis=0)
                    prediction = self.detector.predict(reshaped_image, verbose=0)
                    classification = self.find_class(prediction[0])

                    if classification == 0:
                        self.save_results(ET1_slice.start_time, ET1_slice.end_time, prediction[0][0], 0, result_file_name)
                
                    del ET1_slice, ET2_slice, ET3_slice, img1, img2, img3, E123_inj

                del ET_data1, ET_data2, ET_data3
                gc.collect()
            except Exception as e:
                if self.verbose:
                    logging.error(f"Failed to process {file_suffix}: {str(e)}")
                with open('failed_files.txt', 'a') as f:
                    f.write(f"{file_suffix}\n")

    def process_files(self, data_files) -> None:
        for i in tqdm(range(0, len(data_files), self.batch_size)):
            batch_files = data_files[i:i + self.batch_size]
            self.process_batch(batch_files)
            gc.collect()


def main() -> None:
    # Welcome message :)
    print("*********************************************")
    print("* ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ *")
    print("* ^|^         Welcome to PyMerger         ^|^ *")
    print("* ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ ^|^ *")
    print("* Developed by Wathela Alhassan, et al. 2024. *")
    print("* Email: wathelahamed@gmail.com *")
    print("*********************************************")
    parser = argparse.ArgumentParser(prog="PyMerger",
                                     description='Detect Binary Black Hole mergers from Einstein Telescope data.',
                                     epilog='End of help message.')
    
    parser.add_argument('-r', '--sampling-rate', type=int, choices=[8192, 4096], default=8192,
                        help="Sampling rate of the input data (either 8192 or 4096). Default is 8192.")

    parser.add_argument('-n', '--no-segment', type=int, default=1,
                    help=("Number of data segments to be processed for each detector "
                          "(i.e., number of .gwf files to be processed for each detector). "
                          "Files in the input directory will be sorted, and the first 'n' files up to the specified number of segments will be processed."
                          "Default is 1 which means only the first file from each detector will be scanned."))

    parser.add_argument('-c', '--channels', nargs=3, default=['E1:STRAIN', 'E2:STRAIN', 'E3:STRAIN'],
                    help="List of the THREE channels to be processed. Default is ['E1:STRAIN', 'E2:STRAIN', 'E3:STRAIN'].")

    parser.add_argument('-t', '--threshold', type=float, default=0.1,
                    help="Threshold value for merger detection. A value between 0.1 and 0.5, where a smaller value will result in fewer detections but a lower false positive rate. Default is 0.2. ")

    parser.add_argument('-i', '--input-file-dir', type=str,
                        help="Directory containing the input .gwf files.")

    parser.add_argument('-f', '--output-dir', type=str,
                        help="Directory to store the results.")

    parser.add_argument('--verbose', action='store_true', default=True,
                        help="Enable verbose mode to print update messages. Default is true.")

    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
                       

    if args.input_file_dir is None:
        raise ValueError(f'Path to input data directory was not provided.')
        

    result_path = args.output_dir
    if result_path is None or not os.path.isdir(result_path):
        msg = ("The option --output-dir was not set or the provided path does not exist.")
        logging.info(msg)
        result_path = "PyMerger_result/"
        if not os.path.exists(result_path):
            logging.info(f"A directory named {result_path} will be created in the current directory to store the results.")
            os.makedirs(result_path)
        else:
            logging.info(f"Results will be stored in {result_path} directory.")
    data_path = args.input_file_dir
    verbose = args.verbose

    # Load the pre-trained model
    model_path = model_path = pkg_resources.resource_filename('PyMergers', 'models/pymerger_model.tflite')#".models/pymerger_model.tflite"
    detector = load_model(model_path)

    # choose the sliding-window size based on the given samling rate
    window_size = 2.5 if args.sampling_rate == 8192 else 5 if args.sampling_rate == 4096 else None
    xsize = 91 if args.sampling_rate == 8192 else 182 if args.sampling_rate == 4096 else None
    if verbose:
        logging.info(f"Sliding window set to {window_size} seconds.")

   
    threshold = args.threshold
    channels = args.channels
    num_seg = args.no_segment
    
    process_start_time = time.time()

    # Process 
    processor = ETFileScanner(data_path, num_seg, channels, threshold, window_size, xsize, detector, result_path, verbose)
    processor.scan_files()
    
    process_end_time = time.time()

    # Elapsed time
    elapsed_time = (process_end_time - process_start_time)/60
    if verbose:
        logging.info((f"Done! Processed in: {elapsed_time:.2f} minutes"))  


if __name__=="__main__":
    main()

