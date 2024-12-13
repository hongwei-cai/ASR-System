# Automatic Speech Recognition (ASR) System
Capturing clarity through speech recognition.

## Overview
ASR is a speech recognition system that uses a Deep Neural Network (DNN) to transcribe spoken language into text. The system is trained using the Connectionist Temporal Classification (CTC) loss function, which is well-suited for sequence-to-sequence problems where the alignment between input and output sequences is unknown.

## Project Structure

. ├── pycache/ ├── asr_model.pkl ├── ctc_test.py ├── ctc.py ├── dev_data/ │ ├── feats.1.ark │ ├── feats.10.ark │ └── ... ├── dev_data.json ├── dnn_test.py ├── dnn.py ├── input_generator_test.py ├── input_generator.py ├── LICENSE ├── mnist_input_generator.py ├── mnist_model.pkl ├── mnist.pkl ├── README.md ├── recog_asr.py ├── saved_model.pkl ├── test_data/ │ ├── feats.1.ark │ └── ... ├── test_data.json ├── test_forced_alignment.py ├── test_mnist.py ├── train_asr.py ├── train_data/ │ ├── feats.1.ark │ └── ... ├── train_data.json ├── train_mnist.py └── utils.py

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ASR.git
    cd ASR
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Training
To train the ASR model, run the following command:
```sh
python train_asr.py
```

This will train the model using the training data specified in train_data.json and save the trained model to asr_model.pkl.

## Evaluation
To evaluate the trained model on the development set, run:

```sh
python eval_asr.py
```

This will compute the character error rate (CER) on the development set and print the results.

## Forced Alignment
To test the forced alignment, run:

```sh
python align_asr.py
```

This will use the trained model to perform forced alignment on a test utterance and print the alignment results.

## License
This project is licensed under the terms of the Creative Commons Legal Code.

## Acknowledgements
We would like to thank the contributors and the open-source community for their support and contributions to this project.

This will train the model using the training data specified in train_data.json and save the trained model to asr_model.pkl.

Evaluation
To evaluate the trained model on the development set, run:

python recog_asr.py

This will compute the character error rate (CER) on the development set and print the results.

Forced Alignment
To test the forced alignment, run:

python test_forced_alignment.py

This will use the trained model to perform forced alignment on a test utterance and print the alignment results.

License
This project is licensed under the terms of the Creative Commons Legal Code.

Acknowledgements
We would like to thank the contributors and the open-source community for their support and contributions to this project. ```