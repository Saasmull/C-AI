#include <math.h>

struct Neuron{
    unsigned int n_inputs;
    double* weights;
    double bias;
    double output;
};

typedef struct Neuron Neuron;

Neuron Neuron_init(unsigned int n_inputs){
    Neuron neuron;
    neuron.n_inputs = n_inputs;
    neuron.weights = (double *)malloc(n_inputs * sizeof(double));
    for(unsigned int i = 0; i < n_inputs; i++){
        neuron.weights[i] = 0.0;
    }
    neuron.bias = 0.0;
    neuron.output = 0.0;
    return neuron;
}

double Neuron_sigmod(double x){
    return 1.0 / (1.0 + exp(-x));
}

double Neuron_activate(Neuron *neuron, double *inputs){
    double sum = 0.0;
    for(unsigned int i = 0; i < neuron->n_inputs; i++){
        sum += neuron->weights[i] * inputs[i];
    }
    neuron->output = Neuron_sigmod(sum + neuron->bias);
    return neuron->output;
}


void Neuron_free(Neuron *neuron){
    free(neuron->weights);
}