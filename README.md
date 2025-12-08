## TCNSFormer: Dynamic Global-Local Transformer for Multivariate Time Series Classification

### Install dependencies
```
pip install -r requirements.txt
```

### Model training
```
python main.py --dataset_pos=[dataset_pos] --num_shapelet=[num_shapelet] --window_size=[window_size]
```

Here, [dataset_pos], [num_shapelet],[num_shapelet] and [num_blocks] can be selected as follows:

| Dataset                   | [dataset_pos] | [window_size] | [num_shapelet] | [num_blocks] |
|---------------------------|---------------|---------------|----------------|--------------|
| ArticularyWordRecognition | 0             | 100           | 10             | 2            |
| AtrialFibrillation        | 1             | 100           | 3              | 2            |
| BasicMotions              | 2             | 100           | 10             | 3            |
| CharacterTrajectories     | 3             | 50            | 3              | 1            |
| Cricket                   | 4             | 200           | 30             | 3            |
| DuckDuckGeese             | 5             | 10            | 100            | 2            |
| ERing                     | 6             | 50            | 100            | 3            |
| EigenWorms                | 7             | 10            | 10             | 3            |
| Epilepsy                  | 8             | 20            | 30             | 3            |
| EthanolConcentration      | 9             | 200           | 100            | 2            |
| FaceDetection             | 10            | 10            | 10             | 3            |
| FingerMovements           | 11            | 20            | 30             | 2            |
| HandMovementDirection     | 12            | 200           | 100            | 2            |
| Handwriting               | 13            | 20            | 30             | 2            |   
| Heartbeat                 | 14            | 200           | 100            | 2            |
| InsectWingbeat            | 15            | 10            | 30             | 3            |
| JapaneseVowels            | 16            | 10            | 1              | 1            |
| LSST                      | 17            | 20            | 10             | 2            |
| Libras                    | 18            | 10            | 30             | 3            |
| MotorImagery              | 19            | 100           | 30             | 3            |
| NATOPS                    | 20            | 20            | 1              | 3            |
| PEMS-SF                   | 21            | 50            | 10             | 3            |
| PenDigits                 | 22            | 4             | 10             | 3            |
| PhonemeSpectra            | 23            | 20            | 30             | 3            |
| RacketSports              | 24            | 10            | 10             | 3            |
| SelfRegulationSCP1        | 25            | 100           | 100            | 3            |
| SelfRegulationSCP2        | 26            | 100           | 100            | 3            |
| SpokenArabicDigits        | 27            | 100           | 100            | 3            |
| StandWalkJump             | 28            | 10            | 100            | 3            |
| UWaveGestureLibrary       | 29            | 10            | 10             | 2            |


This repository is for anonymous peer review only. No personal/affiliation information is included.

