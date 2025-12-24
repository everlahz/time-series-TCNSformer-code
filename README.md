## TCNSFormer Multivariate Time Series Classification Method Based on Collaboration of Global Dependence and Local Morphological Features

### Install dependencies
```
pip install -r requirements.txt
```

### Model training
```
python main.py --dataset_pos=[dataset_pos] --num_shapelet=[num_shapelet] --window_size=[window_size] --num_blocks=[num_blocks] --num_layers=[num_layers]
```

Here, [dataset_pos], [num_shapelet],[num_shapelet] and [num_blocks] can be selected as follows:

| Dataset                   | [dataset_pos] | [window_size] | [num_shapelet] | [num_blocks] | [num_layers] |
|---------------------------|---------------|---------------|----------------|--------------|--------------|
| ArticularyWordRecognition | 0             | 100           | 10             | 2            | 2            |
| AtrialFibrillation        | 1             | 100           | 3              | 2            | 2            |
| BasicMotions              | 2             | 100           | 10             | 3            | 2            |
| CharacterTrajectories     | 3             | 50            | 3              | 1            | 1            |
| Cricket                   | 4             | 200           | 30             | 3            | 2            |        
| DuckDuckGeese             | 5             | 10            | 100            | 2            | 2            |        
| ERing                     | 6             | 50            | 100            | 3            | 2            |
| EigenWorms                | 7             | 10            | 10             | 3            | 2            |
| Epilepsy                  | 8             | 20            | 30             | 3            | 3            |
| EthanolConcentration      | 9             | 200           | 100            | 2            | 2            |
| FaceDetection             | 10            | 10            | 10             | 3            | 2            |
| FingerMovements           | 11            | 20            | 30             | 2            | 1            |
| HandMovementDirection     | 12            | 200           | 100            | 2            | 2            |
| Handwriting               | 13            | 20            | 30             | 2            | 1            |   
| Heartbeat                 | 14            | 200           | 100            | 2            | 1            |
| InsectWingbeat            | 15            | 10            | 30             | 3            | 2            |
| JapaneseVowels            | 16            | 10            | 1              | 1            | 2            |
| LSST                      | 17            | 20            | 10             | 2            | 1            |
| Libras                    | 18            | 10            | 30             | 3            | 2            |
| MotorImagery              | 19            | 100           | 30             | 3            | 2            |
| NATOPS                    | 20            | 20            | 1              | 3            | 3            |
| PEMS-SF                   | 21            | 50            | 10             | 3            | 1            |
| PenDigits                 | 22            | 4             | 10             | 3            | 2            |
| PhonemeSpectra            | 23            | 20            | 30             | 3            | 1            |
| RacketSports              | 24            | 10            | 10             | 3            | 1            |
| SelfRegulationSCP1        | 25            | 100           | 100            | 3            | 1            |
| SelfRegulationSCP2        | 26            | 100           | 100            | 3            | 2            |
| SpokenArabicDigits        | 27            | 100           | 100            | 3            | 2            |
| StandWalkJump             | 28            | 10            | 100            | 3            | 1            |
| UWaveGestureLibrary       | 29            | 10            | 10             | 2            | 1            |


This repository is for anonymous peer review only. No personal/affiliation information is included.

