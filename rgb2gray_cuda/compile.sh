#!/bin/bash

nvcc grey.cu helpers.cu `pkg-config --cflags --libs opencv`
