## System Requirements

### System Hardware:

* CPU: Intel(R) Core(TM) i7-13650HX (2.60 GHz) 
* RAM: 16 GB DDR5
* GPU: NVIDIA RTX 4050 (8 GB)

### Operating System

* Windows 11

### Software Requirements
* VScode

### Python

* Python 3.10.19

### Main Libraries

* absl-py
* torch==2.9.0
* torchvision==0.24.0
* torchaudio==2.9.0
* jax[cpu]
* scipy
* numpy
* tabulate
* safetensors
* kagglehub
* transformers
* multipledispatch
* torchdiffeq
* pandas
* numpy
* scikit-learn
* openpyxl
* matplotlib
* pathlib
* codecarbon
* psutil

## Running Experiments 

### Folder Contents

* **Data_for_Main**: Data
* **Model train and Figure**: Teacher models training
* **Model Compression All**: DLNet Compression
* **Docker Implementation**: TFLite Conversion via Docker
* **Prototype_Deployment**: Arduino deployment and Web Interface

### Expected Runtime

* Training (200 epochs): 2 to 5 mins
* Compression: 15 mins
* Inference: 0.5 to 2 ms (21 ms on Arduino Nano)

### Reproducibility

* To reproduce the results, use the saved models from the folders "Model train anf Figure" and "Model Compression All" for teacher and compressed models (using DLNet).


## Docker:

TFLite Conversion and Reproducible Model testing
* Tflite_convert.py - further compress the compressed Pytorch models to Tflite models.
* Tflite_test.py - test TFLite models.
* Pytorch_test.py - test teacher and compressed Pytorch models.

(All the necessary files are available in the folder "Docker Implementation". The requirements are given in "requirements.txt")


## Arduino Deployment (Optional):

This is only a prototype, by direclty feeding the input data; the external data are also collected for reference purposes. (Refer "Circuit Connection.png" for the circuit setup)

### Hardware:

* Arduino Nano 33 BLE Sense Rev 2
* INA219 module
* DS18B20 temperature sensor 
* I2C-enabled LCD1602 RGB from DFRobot
* IRF18650 LiFePO4 Battery


### Software:

* Arduino IDE
* Web Interface - VSCode, Google Chrome/ Microsoft Edge

### Programming languages:
* C++
* Html

### Libraries:
* Arduino
* Wire
* ArduinoBLE
* DHT
* INA219_WE
* DFRobot_RGBLCD1602
* Chirale_TensorFlowLite

### Instructions

The code and model for deployment are found in the folder "Prototype_Deployment"

