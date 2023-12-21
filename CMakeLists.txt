cmake_minimum_required(VERSION 3.25)
project(C_AI C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(C_AI
        main.c
        Neuron.h
        NeuroNet.h)
