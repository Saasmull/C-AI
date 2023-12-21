#include "stdio.h"

#include "NeuroNet.h"

struct DArray{
    double *data;
    int size;
};

typedef struct DArray DArray;

DArray DArray_init(int size){
    DArray arr;
    arr.data = (double*)malloc(sizeof(double) * size);
    arr.size = size;
    return arr;
}

void DArray_free(DArray *arr){
    free(arr->data);
}

int main(){

    // inputs
    DArray inputA = DArray_init(2);
    inputA.data[0] = 0.0;
    inputA.data[1] = 0.0;
    DArray inputB = DArray_init(2);
    inputB.data[0] = 0.0;
    inputB.data[1] = 1.0;
    DArray inputC = DArray_init(2);
    inputC.data[0] = 1.0;
    inputC.data[1] = 0.0;
    DArray inputD = DArray_init(2);
    inputD.data[0] = 1.0;
    inputD.data[1] = 1.0;

    // outputs
    DArray outputA = DArray_init(1);
    outputA.data[0] = 0.0;
    DArray outputB = DArray_init(1);
    outputB.data[0] = 1.0;
    DArray outputC = DArray_init(1);
    outputC.data[0] = 1.0;
    DArray outputD = DArray_init(1);
    outputD.data[0] = 1.0;

    // Train the AI
    NeuroNet net = NeuroNet_init(2, 2, 1);
    for(unsigned int i = 0; i < 100000; i++){
        NeuroNet_train(&net, inputA.data, outputA.data);
        NeuroNet_train(&net, inputB.data, outputB.data);
        NeuroNet_train(&net, inputC.data, outputC.data);
        NeuroNet_train(&net, inputD.data, outputD.data);
    }

    DArray resultA = DArray_init(1);
    resultA.data = NeuroNet_predict(&net, inputA.data);
    DArray resultB = DArray_init(1);
    resultB.data = NeuroNet_predict(&net, inputB.data);
    DArray resultC = DArray_init(1);
    resultC.data = NeuroNet_predict(&net, inputC.data);
    DArray resultD = DArray_init(1);
    resultD.data = NeuroNet_predict(&net, inputD.data);

    printf("0, 0 -> 0 -> %f\n", resultA.data[0]);
    printf("0, 1 -> 1 -> %f\n", resultB.data[0]);
    printf("1, 0 -> 1 -> %f\n", resultC.data[0]);
    printf("1, 1 -> 1 -> %f\n", resultD.data[0]);

    DArray_free(&inputA);
    DArray_free(&inputB);
    DArray_free(&inputC);
    DArray_free(&inputD);
    DArray_free(&outputA);
    DArray_free(&outputB);
    DArray_free(&outputC);
    DArray_free(&outputD);
    NeuroNet_free(&net);
    DArray_free(&resultA);
    DArray_free(&resultB);
    DArray_free(&resultC);
    DArray_free(&resultD);
    return 0;
}

