from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model():
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(40,100,1)))
    model.add(MaxPooling2D((2,2)))

    # Conv Block 2
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    # Flatten + Dense
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    # Output
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# Test build
if __name__ == "__main__":
    model = build_model()
    with open("model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))