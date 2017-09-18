#!/bin/bash

nvcc blur.cu helpers.cu `pkg-config --cflags --libs opencv`
