# Installation

# Usage

Record your gestures with `record.py`. Record at least 20 samples per gesture. They will be in folders named 'Up', etc. Move these to the Data folder 'Data/Up'.

Train the svm using `train.py`. Update this file with the gesture labels you recorded.

```
lookup = {
    "Up": 0,
    "Down": 1,
    "Left": 2,
    "Right": 3,
    "Neutral": 4,
    "Clap": 5,
    "Fetch": 6,
}
```

Control the drone with `main.py`. You can change the sensitivity with `confidence_threshold` in the file.
