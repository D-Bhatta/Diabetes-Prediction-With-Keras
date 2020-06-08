# Diabetes-Prediction-With-Keras

Diabetes prediction with keras, by following the [tutorial](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)

## Sections

Load Data
Define Keras Model
Compile Keras Model
Fit Keras Model
Evaluate Keras Model
Tie It All Together
Make Predictions

## Notes

1. Write as much code as you can in CPU before running it all in a GPU
2. Install tensorflow and keras

    ```python
    !pip install tensorflow
    !pip install tensorflow-gpu
    !pip install keras
    ```

3. Check versions

    ```python
    print(tf.__version__)
    ```

4. Check for presence of GPU. Turn off GPU when not in use.

    ```python
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
    raise SystemError("GPU device not found")
    print(f'Found GPU at: {device_name}')
    ```

5. Download dataset using `!wget -cq` **link**

    ```python
    !wget -cq https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
    ```

6. Import modules, such as joblib and keras layers and models

    ```python
    import tensorflow as tf
    from numpy import loadtxt
    from keras.models import Sequential
    from keras.layers import Dense
    from joblib import load
    from joblib import dump
    ```

7. Load dataset

    ```python
    dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
    x = dataset[:,0:8]
    y = dataset[:,8]
    ```

8. Define the model.
   1. Create a sequential class object
   2. First layer is the input layer, so nodes={input_dim+4}. Set the input dimension there as well. Also, set activation function as Relu
   3. Define the second layer as nodes={input_dim+4}
   4. Define output layer with sigmoid function

    ```python
    model = Sequential()
    model.add(Dense(12,input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    ```

9. Compile the model using loss function, optimizer, and metrics to evaluate for each epoch

    ```python
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ```

10. Train model on GPU

    ```python
    model.fit(x,y, epochs=150, batch_size=10)
    ```

11. Evaluate model and print accuracy

    ```python
    _, accuracy = model.evaluate(x,y)
    print(f'Accuracy: {accuracy*100}')
    ```

12. Make predictions using model. Without a test set this just tests recall

    ```python
    y_pred = model.predict(x)
    y_pred = [1 if i>=0.5 else 0 for i in y_pred]
    ```

13. Calculate statistics

    ```python
    total = 0
    correct = 0
    wrong = 0
    for i in range(len(y_pred)):
    total = total + 1
    if y_pred[i] == y[i]:
        correct = correct + 1
    else:
        wrong = wrong + 1
    ```

14. Pipeline model
    1. Dump model
    2. Load model
    3. Evaluate model and print accuracy
    4. Make predictions using model. Without a test set this just tests recall
    5. Check if they match previous stats in step **13**.

    ```python
    dump(model, 'model.sav')
    test_model = load('model.sav')
    _, accuracy = test_model.evaluate(x,y)
    print(f'Accuracy: {accuracy*100}')
    test_y_pred = test_model.predict(x)
    test_y_pred = [1 if i>=0.5 else 0 for i in test_y_pred]
    test_total = 0
    test_correct = 0
    test_wrong = 0
    for i in range(len(test_y_pred)):
    test_total = test_total + 1
    if test_y_pred[i] == y[i]:
        test_correct = test_correct + 1
    else:
        test_wrong = test_wrong + 1
    print(f"Loaded Model: \n Total: {test_total} \n Correct: {test_correct} \n Wrong: {test_wrong}")
    ```

## Output

```markdown
Accuracy: 76.953125
Loaded Model:
 Total: 768
 Correct: 591
 Wrong: 177
```

## Project status

Project has been completed successfully
