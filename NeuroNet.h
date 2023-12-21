
#include <stdlib.h>

#include "Neuron.h"

struct NeuroNet{
    unsigned int inputSize;
    unsigned int hiddenSize;
    unsigned int outputSize;
    Neuron *inputLayer;
    Neuron *hiddenLayer;
};

typedef struct NeuroNet NeuroNet;

NeuroNet NeuroNet_init(unsigned int inputSize, unsigned int hiddenSize, unsigned int outputSize){
    NeuroNet net;
    net.inputSize = inputSize;
    net.hiddenSize = hiddenSize;
    net.outputSize = outputSize;
    net.inputLayer = (Neuron *)malloc(net.inputSize * sizeof(Neuron));
    for(unsigned int i = 0; i < inputSize; i++){
        net.inputLayer[i] = Neuron_init(inputSize);
    }
    net.hiddenLayer = (Neuron *)malloc(net.hiddenSize * sizeof(Neuron));
    for(unsigned int i = 0; i < hiddenSize; i++){
        net.hiddenLayer[i] = Neuron_init(hiddenSize);
    }
    return net;
}

double* NeuroNet_predict(NeuroNet *net, double *inputs){
    // Forward propagation
    double *hiddenOutputs = (double *)malloc(net->hiddenSize * sizeof(double));
    for(unsigned int i = 0; i < net->hiddenSize; i++){
        hiddenOutputs[i] = Neuron_activate(&net->hiddenLayer[i], inputs);
    }
    double *outputs = (double *)malloc(net->outputSize * sizeof(double));
    for(unsigned int i = 0; i < net->outputSize; i++){
        outputs[i] = Neuron_activate(&net->inputLayer[i], hiddenOutputs);
    }
    free(hiddenOutputs);
    return outputs;
}

void NeuroNet_train(NeuroNet *net, double *inputs, double *targets){
    double learningRate = 0.1;
    unsigned int epochs = 1;
    for(unsigned int epoch = 0;epoch < epochs;epoch++){
        // Forward propagation
        double *hiddenOutputs = (double *)malloc(net->hiddenSize * sizeof(double));
        for(unsigned int i = 0; i < net->hiddenSize; i++){
            hiddenOutputs[i] = Neuron_activate(&net->hiddenLayer[i], inputs);
        }
        double *outputs = (double *)malloc(net->outputSize * sizeof(double));
        for(unsigned int i = 0; i < net->outputSize; i++){
            outputs[i] = Neuron_activate(&net->inputLayer[i], hiddenOutputs);
        }

        // Calculate output errors
        double *outputErrors = (double *)malloc(net->outputSize * sizeof(double));
        for(unsigned int i = 0; i < net->outputSize; i++){
            outputErrors[i] = targets[i] - outputs[i];
        }

        // Back propagation for output layer
        for(unsigned int i = 0; i < net->outputSize; i++){
            for(unsigned int j = 0; j < net->hiddenSize; j++){
                double gradient = outputs[i] * (1.0 - outputs[i]) * hiddenOutputs[j];
                net->inputLayer[i].weights[j] += learningRate * outputErrors[i] * outputs[i] * gradient;
            }
            net->inputLayer[i].bias += learningRate * outputErrors[i] * outputs[i] * (1.0 - outputs[i]);
        }

        // Back propagation for hidden layer
        for(unsigned int i = 0; i < net->hiddenSize; i++){
            double errorSum = 0.0;
            for(unsigned int j = 0; j < net->outputSize; j++){
                errorSum += outputErrors[j] * net->inputLayer[j].weights[i];
            }
            for(unsigned int j = 0; j < net->inputSize; j++){
                double gradient = hiddenOutputs[i] * (1.0 - hiddenOutputs[i]) * inputs[j];
                net->hiddenLayer[i].weights[j] += learningRate * errorSum * hiddenOutputs[i] * gradient;
            }
            net->hiddenLayer[i].bias += learningRate * errorSum * hiddenOutputs[i] * (1.0 - hiddenOutputs[i]);
        }

        // Free memory
        free(hiddenOutputs);
        free(outputs);
        free(outputErrors);
    }
}



void NeuroNet_free(NeuroNet *net){
    for(unsigned int i = 0; i < net->inputSize; i++){
        Neuron_free(&net->inputLayer[i]);
    }
    free(net->inputLayer);
    for(unsigned int i = 0; i < net->hiddenSize; i++){
        Neuron_free(&net->hiddenLayer[i]);
    }
    free(net->hiddenLayer);
}