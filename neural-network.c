#include <stdlib.h>
#include <stdio.h>
#include <math.h>   

//Training a neural network with 2 neurons to learn the XOR function

#define numInputs 2
#define numHiddenNodes 4
#define numOutputs 1
#define numTrainingSets 4

double initWheigts(){return ((double)rand()) / ((double)RAND_MAX);}

double sigmoid(double x){return 1 / (1 + exp(-x));}
double derivateSigmoid(double x){return x * (1-x);}     

void shuffle(int  *array, size_t n){ 
    if(n>1){
        size_t i;
        for(i = 0; i<n - 1; i++){
            size_t j = i + rand() / (RAND_MAX / (n-i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int main(void){
    const double learningRange = 0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    double training_inputs[numTrainingSets][numInputs] = {{0.0f, 0.0f},{1.0f, 0.0f},{0.0f, 0.1f},{1.0f, 1.0f}};

    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},{1.0f},{1.0f},{0.0f}};

    for(int i = 0; i<numInputs; i++){
        for(int j = 0; i<numHiddenNodes; j++){
            hiddenWeights[i][j] = initWheigts();
        }
    }

    for(int i = 0; i<numHiddenNodes; i++){
        for(int j = 0; i<numOutputs; j++){
            outputWeights[i][j] = initWheigts();
        }
    }

    for(int i = 0; i<numOutputs; i++){
        outputLayerBias[i] = initWheigts();
    }

    int trainingSetOrder[] = {0,1,2,3};
    
    int numberEpochs = 100000;

    for(int epoch = 0; epoch<numberEpochs; epoch++){
        shuffle(trainingSetOrder,numTrainingSets);
        for(int x = 0; x<numTrainingSets; x++){
            int i = trainingSetOrder[x];

            for(int j =0; j<numHiddenNodes; j++){
                double activate = hiddenLayerBias[j];
                for(int k =0; k<numInputs; k++){
                   activate += training_inputs[i][k] * hiddenWeights[k][j];
                }
            hiddenLayer[j] = sigmoid(activate);
            }
            
            for(int j =0; j<numOutputs; j++){
                double activate = outputLayerBias[j];
                for(int k =0; k<numHiddenNodes; k++){
                   activate += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activate);
            }
            
            printf("INPUT: %g %g \t OUTPUT: %g \t PREDICTED OUTPUT: %g \n",training_inputs[i][0],training_inputs[i][1],outputLayer[0], training_outputs[i][0]);

            double deltaOutput[numOutputs];

            for (int j = 0; j<numOutputs; j++){
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * derivateSigmoid(outputLayer[j]);
            }   

            double deltaHidden[numHiddenNodes];
            for(int j = 0; j<numHiddenNodes; j++){
                double error = 0.0f;
                for(int k = 0; k<numOutputs; k++){
                    error +=deltaOutput[k]*outputWeights[j][k];
                    
                }
                deltaHidden[j] = error*derivateSigmoid(outputLayer[j]);
            }

            for(int j =0; j<numOutputs; j++){
                outputLayerBias[j] += deltaOutput[j] * learningRange;
                for(int k =0; k<numHiddenNodes; k++){
                    outputWeights[k][j] += deltaOutput[j] * hiddenLayer[k] * learningRange;
                }
            }

            for(int j =0; j<numHiddenNodes; j++){
                hiddenLayerBias[j] += deltaHidden[j] * learningRange;
                for(int k =0; k<numInputs; k++){
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * learningRange;
                }
            }

        }
        
    }
    fputs("Final Hidden Weights\n", stdout);
        for (int j = 0; j<numHiddenNodes; j++){
            fputs("[",stdout);
            for (int k = 0; k<numInputs; k++){
                printf("%f ", hiddenWeights[k][j]);
            }
            fputs("]", stdout);
        }

        fputs("\nFinal Hiden Biases\n[", stdout);
        for(int j =0; j<numHiddenNodes; j++){
            printf("%f ", hiddenLayerBias[j]);
        }
        fputs("]", stdout);
        
        fputs("\nFinal Hidden Outputs\n", stdout);
        for (int j = 0; j<numOutputs; j++){
            fputs("[",stdout);
            for (int k = 0; k<numHiddenNodes; k++){
                printf("%f ", outputWeights[k][j]);
            }
            fputs("]\n", stdout);
        }

        fputs("Final Output Biases\n[", stdout);
        for(int j =0; j<numOutputs; j++){
            printf("%f ", outputLayerBias[j]);
        }
        fputs("]\n", stdout);
    return 0;
}
