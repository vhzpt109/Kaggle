import pandas as pd
import numpy as np

from tensorflow.keras.utils import to_categorical
from models import LeNet

if __name__ == "__main__":
    data_path = "D:/Kaggle/Digit Recognizer/"
    train = pd.read_csv(data_path + "train.csv")
    test = pd.read_csv(data_path + "test.csv")

    train_input = train.drop(columns="label").to_numpy()
    train_label = train["label"].to_numpy()

    train_input = train_input / 255.
    train_input = train_input.reshape(-1, 28, 28, 1)

    train_label = to_categorical(train_label)

    model = LeNet()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_input, train_label, batch_size=128, epochs=10, validation_split=0.2)

    test_input = test.to_numpy()
    test_input = test_input / 255.
    test_input = test_input.reshape(-1, 28, 28, 1)

    prediction = model.predict(test_input)
    prediction = np.argmax(prediction, axis=1)

    submission = pd.DataFrame({
        "ImageId": [i for i in range(1, len(test) + 1)],
        "Label": prediction
    })

    submission.to_csv('submission.csv', index=False)