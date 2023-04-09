import time
import tkinter
from tkinter import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import embed_plot
import math



SIZE = 5
def initMatrix(size):
    return [[0.0 for i in range(size)] for i in range(size)]
def editMatrixIndex(matrix, row, col, val):
    #matrix = [[...],
    #          [...],...]
    if row < len(matrix) and col < len(matrix[row]):
        matrix[row][col] = val
        return matrix
    else: return matrix
matrix1 = np.array(initMatrix(SIZE))
print(editMatrixIndex(matrix1, 3, 2, .5))
print(editMatrixIndex(matrix1, 4, 1, .9))
print(editMatrixIndex(matrix1, 2, 4, .2))
print(editMatrixIndex(matrix1, 0, 2, .4))

model = keras.Sequential()
#model.add(keras.layers.Flatten(input_shape=(SIZE, )))
layer2 = keras.layers.Dense(25, activation="relu")
model.add(layer2)
layer3 = keras.layers.Dense(5, activation="sigmoid")
model.add(layer3)
layer1 = keras.layers.Dense(1, activation="sigmoid")
model.add(layer1)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])



window = Tk()
window.resizable(False, False)
canvas = Canvas(window,
                bg="black",
                highlightbackground="blue",
                highlightthickness=1,
                width=500,
                height=500)
canvas.pack()


(plot1, plotCanvas1) = embed_plot.embedPlot(window, 200, 300, 2, [], [] )
(plot2, plotCanvas2) = embed_plot.embedPlot(window, 0, 300, 2, [], [] )
def plotMatrix(matrix, plot, plotCanvas):
    (numRows, numCols) = np.shape(matrix)
    inputX = []
    inputY = []
    points = []
    created = []
    for row in range(numRows):
        for col in range(numCols):
            inputX.append(col)
            inputY.append(row)
            points.append(matrix[row][col])
    points = np.array(points)

    color = np.sqrt((points**2))/np.sqrt(2.0)

    rgb = plt.get_cmap('jet')(color)
    inputX.reverse()
    inputY.reverse()

    newPoint = plot.scatter(inputX, inputY, color=rgb)
    plt.show()
    created.append(newPoint)
    plotCanvas.draw()
    canvas.update()
    
    for obj in created:
        obj.remove()
    
    

def plotKerasLayer(layer):
    w = layer.get_weights()[0]
    print("Weights: ", w)
    plotMatrix(w)


while True:
    time.sleep(.01)
    #matrixOut1 = model.predict(matrix1)
    # plotKerasLayer(layer1)
    matrixOut1 = model.predict(matrix1)
    print("Predict: ", matrixOut1)
    plotMatrix(matrixOut1, plot1, plotCanvas1)
    trainOut = tf.constant([[50.0,0.,0.,0.,0.]])

    plotMatrix(trainOut, plot2, plotCanvas2)
    model.fit(tf.constant([matrix1]), trainOut, epochs=10, verbose=0)
    
    window.update()