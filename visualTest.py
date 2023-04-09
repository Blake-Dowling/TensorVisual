import time
import tkinter
from tkinter import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import visual.embed_plot as embed_plot
import math
import visual.visual as visual
from visual.visual import *



if __name__ == "__main__":
    SIZE = 5
    ####################Initializing Test Matrix####################
    def initMatrix(size):
        return [[0.0 for i in range(size)] for i in range(size)]
    def editMatrixIndex(matrix, row, col, val):
        #matrix = [[...],
        #          [...],...]
        if row < len(matrix) and col < len(matrix[row]):
            matrix[row][col] = val
            return matrix
        else: return matrix
    matrix1 = initMatrix(SIZE)
    print(editMatrixIndex(matrix1, 3, 2, .5))
    print(editMatrixIndex(matrix1, 4, 1, .9))
    print(editMatrixIndex(matrix1, 2, 4, .2))
    print(editMatrixIndex(matrix1, 0, 2, .4))
    ####################Initializing Keras Model####################
    model = keras.Sequential()
    layer0 = keras.layers.Flatten(input_shape=(SIZE,SIZE))
    model.add(layer0)
    layer2 = keras.layers.Dense(25, activation="relu")
    model.add(layer2)
    layer1 = keras.layers.Dense(5, activation="softmax")
    model.add(layer1)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


    ####################Initializing Tkinter Window####################
    window = Tk()
    window.resizable(False, False)
    canvas = Canvas(window,
                    bg="black",
                    highlightbackground="blue",
                    highlightthickness=1,
                    width=500,
                    height=500)
    canvas.pack()
    ####################Animation Loop####################
    matrixPlot1 = visual.MatrixPlot(window, 0, 300)
    matrixPlot2 = visual.MatrixPlot(window, 200, 300)
    while True:
        time.sleep(.01)
        #matrixOut1 = model.predict(matrix1)
        matrixPlot2.plotKerasLayer(layer1)
        trainIn = [matrix1]
        matrixOut1 = model.predict(trainIn)
        print("Predict: ", matrixOut1)
        matrixPlot1.plotMatrix(matrixOut1)
        trainOut = 3.0

        #print(matrix1)
        #plotMatrix(matrix1, plot2, plotCanvas2)
        
        # print(trainIn.shape, trainOut.shape)

        model.fit(trainIn, [trainOut], epochs=10, verbose=0)
        
        window.update()



