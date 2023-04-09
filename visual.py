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

model = keras.Sequential()
#model.add(keras.layers.Flatten(input_shape=(SIZE, )))
model.add(keras.layers.Dense(4, activation="relu"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

matrixOut1 = model.predict(matrix1)


window = Tk()
window.resizable(False, False)
canvas = Canvas(window,
                bg="black",
                highlightbackground="blue",
                highlightthickness=1,
                width=500,
                height=500)
canvas.pack()


(plot, plotCanvas) = embed_plot.embedPlot(window, 0, 300, 2, [], [] )
def plotMatrix(matrix):
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
    canvas.update()
    for obj in created:
        canvas.delete(obj)


while True:
    time.sleep(1)
    matrixOut1 = model.predict(matrix1)
    print(matrixOut1)
    plotMatrix(matrixOut1)
    window.update()