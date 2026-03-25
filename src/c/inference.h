#ifndef TINY_ML_INFERENCE_H
#define TINY_ML_INFERENCE_H

#define MAX_LAYERS   10
#define MAX_FEATURES 784

typedef struct {
    int   num_layers;
    int   layer_sizes[MAX_LAYERS];
    float *weights[MAX_LAYERS];
    float *biases[MAX_LAYERS];
    int   has_scaler;
    float scaler_mean[MAX_FEATURES];
    float scaler_std[MAX_FEATURES];
} Network;

int load_weights(Network *net, const char *filename);
void free_network(Network *net);
void linear(float *in, float *W, float *b, float *out, int rows, int cols);
void relu(float *x, int size);
void softmax(float *x, int size);
int predict(Network *net, float *input, int verbose);
void benchmark(Network *net, float *sample_input, int iterations);

#endif
