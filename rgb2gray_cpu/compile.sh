#!/bin/bash

g++ grey.cpp `pkg-config --cflags --libs opencv`
