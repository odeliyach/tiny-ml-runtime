#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE  4
#define HIDDEN_SIZE 8
#define OUTPUT_SIZE 3

typedef struct {
    float w1[HIDDEN_SIZE][INPUT_SIZE];
    float b1[HIDDEN_SIZE];
    float w2[OUTPUT_SIZE][HIDDEN_SIZE];
    float b2[OUTPUT_SIZE];
    float scaler_mean[INPUT_SIZE];
    float scaler_std[INPUT_SIZE];
} Network;

int load_weights(Network *net, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("Error: could not open %s\n", filename);
        return 0;
    }

    fread(net->w1,          sizeof(float), HIDDEN_SIZE * INPUT_SIZE,  f);
    fread(net->b1,          sizeof(float), HIDDEN_SIZE,               f);
    fread(net->w2,          sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
    fread(net->b2,          sizeof(float), OUTPUT_SIZE,               f);
    fread(net->scaler_mean, sizeof(float), INPUT_SIZE,                f);
    fread(net->scaler_std,  sizeof(float), INPUT_SIZE,                f);

    fclose(f);
    return 1;
}

void linear(float *in, float *W, float *b, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        out[i] = b[i];
        for (int j = 0; j < cols; j++) {
            out[i] += W[i * cols + j] * in[j];
        }
    }
}

void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];
    
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

int predict(Network *net, float *input, int verbose) {
    // Normalize
    float normalized[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
        normalized[i] = (input[i] - net->scaler_mean[i]) / net->scaler_std[i];
    }

    // Layer 1
    float hidden[HIDDEN_SIZE];
    linear(normalized, (float*)net->w1, net->b1, hidden, HIDDEN_SIZE, INPUT_SIZE);
    relu(hidden, HIDDEN_SIZE);

    // Layer 2
    float output[OUTPUT_SIZE];
    linear(hidden, (float*)net->w2, net->b2, output, OUTPUT_SIZE, HIDDEN_SIZE);
    softmax(output, OUTPUT_SIZE);

    // Print probabilities
    if (verbose)
        printf("Probabilities: [%.3f, %.3f, %.3f]\n", output[0], output[1], output[2]);
    // Argmax
    int best = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > output[best]) best = i;
    }
    return best;
}
void benchmark(Network *net) {
    float flower[] = {5.1, 3.5, 1.4, 0.2};
    int iterations = 1000000;
    
    clock_t start = clock();
    
    for (int i = 0; i < iterations; i++) {
        predict(net, flower, 0);
    }
    
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    double per_second = iterations / seconds;
    
    printf("\n--- Benchmark ---\n");
    printf("Iterations:        %d\n", iterations);
    printf("Time:              %.3f seconds\n", seconds);
    printf("Predictions/sec:   %.0f\n", per_second);
}
int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    Network net;
    if (!load_weights(&net, "weights.bin")) return 1;
    printf("Weights loaded successfully!\n\n");

    // Test flowers
    char *labels[] = {"Setosa", "Versicolor", "Virginica"};

    float flowers[3][4] = {
        {5.1, 3.5, 1.4, 0.2},   // Setosa
        {6.0, 2.9, 4.5, 1.5},   // Versicolor
        {6.3, 3.3, 6.0, 2.5},   // Virginica
    };

    for (int i = 0; i < 3; i++) {
        printf("Flower %d: [%.1f, %.1f, %.1f, %.1f]\n", 
               i+1, flowers[i][0], flowers[i][1], flowers[i][2], flowers[i][3]);
        int pred = predict(&net, flowers[i], 1);
        printf("Predicted: %s (class %d)\n\n", labels[pred], pred);
    }

    benchmark(&net);
    return 0;
}
