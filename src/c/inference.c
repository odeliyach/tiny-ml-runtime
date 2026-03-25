#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "inference.h"

/**
 * Loads neural network weights from a binary file.
 *
 * File format:
 *   [num_layers: int32]
 *   [layer_sizes: int32 x num_layers]
 *   [W0, b0, W1, b1, ...: float32 arrays]
 *   [scaler_mean, scaler_std: float32 arrays] (optional)
 *
 * @param net      Pointer to Network struct to populate
 * @param filename Path to binary weights file
 * @return         1 on success, 0 on failure
 */
int load_weights(Network *net, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { printf("Error: could not open %s\n", filename); return 0; }
    memset(net, 0, sizeof(*net));

    // First: read architecture
    // e.g. [3, 4, 8, 3] means 3 layers of sizes 4, 8, 3
    if (fread(&net->num_layers, sizeof(int), 1, f) != 1) {
        fclose(f);
        printf("Error: failed to read num_layers\n");
        return 0;
    }
    if (fread(net->layer_sizes, sizeof(int), net->num_layers, f) != (size_t)net->num_layers) {
        fclose(f);
        printf("Error: failed to read layer_sizes\n");
        return 0;
    }

    // For each pair of adjacent layers, load weights and biases
    int layers_allocated = 0;
    for (int i = 0; i < net->num_layers - 1; i++) {
        int rows = net->layer_sizes[i + 1];
        int cols = net->layer_sizes[i];

        // malloc because size is unknown at compile time
        net->weights[i] = malloc(rows * cols * sizeof(float));
        net->biases[i]  = malloc(rows * sizeof(float));

        if (!net->weights[i] || !net->biases[i]) {
            fclose(f);
            printf("Error: memory allocation failed\n");
            goto cleanup_fail;
        }

        if (fread(net->weights[i], sizeof(float), rows * cols, f) != (size_t)(rows * cols)) {
            fclose(f);
            printf("Error: failed to read weights for layer %d\n", i);
            goto cleanup_fail;
        }
        if (fread(net->biases[i], sizeof(float), rows, f) != (size_t)rows) {
            fclose(f);
            printf("Error: failed to read biases for layer %d\n", i);
            goto cleanup_fail;
        }
        layers_allocated++;
    }

    // Try to load scaler — if present in file (Iris has it, MNIST doesn't)
    int input_size = net->layer_sizes[0];
    size_t n = fread(net->scaler_mean, sizeof(float), input_size, f);
    if (n == (size_t)input_size) {
        if (fread(net->scaler_std, sizeof(float), input_size, f) != (size_t)input_size) {
            fclose(f);
            printf("Error: failed to read scaler_std\n");
            goto cleanup_fail;
        }
        net->has_scaler = 1;
    } else {
        net->has_scaler = 0;
    }

    fclose(f);
    return 1;

cleanup_fail:
    for (int j = 0; j < layers_allocated; j++) {
        free(net->weights[j]);
        free(net->biases[j]);
        net->weights[j] = NULL;
        net->biases[j] = NULL;
    }
    net->num_layers = 0;
    return 0;
}

// Free all dynamically allocated weight memory
void free_network(Network *net) {
    for (int i = 0; i < net->num_layers - 1; i++) {
        free(net->weights[i]);
        free(net->biases[i]);
    }
}

/**
 * Performs linear transformation: out = W @ in + b
 *
 * W is stored as a flat 1D array in row-major order.
 * In Python: out = W @ in + b (one line)
 * In C we have no 2D arrays, so we do it manually.
 *
 * @param in   Input vector (size: cols)
 * @param W    Weight matrix, row-major flat array (size: rows * cols)
 * @param b    Bias vector (size: rows)
 * @param out  Output vector (size: rows)
 * @param rows Number of output neurons
 * @param cols Number of input features
 */
void linear(float *in, float *W, float *b, float *out, int rows, int cols) {

    // Loop over each neuron in the output layer
    for (int i = 0; i < rows; i++) {

        // Start with the bias for this neuron (equivalent to the +b part)
        out[i] = b[i];

        // Loop over each input and accumulate the weighted sum:
        // out[i] += W[i][0]*in[0] + W[i][1]*in[1] + ...
        for (int j = 0; j < cols; j++) {

            // W[i * cols + j] is how we access W[i][j] in a flat array
            // because row i starts at position i*cols in memory
            // In C there are only 1D arrays — 2D matrices are stored flat
            out[i] += W[i * cols + j] * in[j];
        }
    }
}

/**
 * Applies ReLU activation function in-place: f(x) = max(0, x)
 *
 * WHY: without non-linearity, stacking linear layers is pointless —
 * W2 @ (W1 @ x) = (W2@W1) @ x, which collapses to one linear layer.
 * ReLU breaks this, allowing the network to learn complex patterns.
 *
 * @param x    Array to modify in-place
 * @param size Number of elements
 */
void relu(float *x, int size) {
    for (int i = 0; i < size; i++)
        if (x[i] < 0) x[i] = 0;
}

/**
 * Applies softmax activation in-place: p_i = exp(x_i) / sum(exp(x_j))
 *
 * Converts raw scores to probabilities that sum to 1.0.
 * Only used at the final layer.
 *
 * Uses max-subtraction trick for numerical stability:
 * - Without: exp(1000) = infinity (overflow)
 * - With:    exp(1000-1000) = exp(0) = 1.0 (safe)
 *
 * @param x    Array to modify in-place
 * @param size Number of elements
 */
void softmax(float *x, int size) {

    // Step 1: find the maximum value — for numerical stability only
    // This is NOT the predicted class
    // Without this: expf(1000) = infinity (float overflow)
    // With this:    expf(1000-1000) = expf(0) = 1.0 (safe)
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];

    // Step 2: compute exp(x - max) and sum
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // Step 3: divide by sum → all values now sum to 1.0
    for (int i = 0; i < size; i++) x[i] /= sum;

    // Note: the PREDICTED CLASS is found separately using argmax
    // That's a different max — not this one!
}

/**
 * Runs forward pass through the network and returns predicted class.
 *
 * Generic implementation — works for ANY architecture.
 * Uses two buffers (buf_a, buf_b) and ping-pongs between them:
 *   layer 0: read buf_a → write buf_b
 *   layer 1: read buf_b → write buf_a  ...etc
 * This avoids allocating a new buffer per layer.
 *
 * @param net     Pointer to loaded Network struct
 * @param input   Input feature vector (size: net->layer_sizes[0])
 * @param verbose If 1, prints probabilities to stdout
 * @return        Predicted class index (0-based)
 */
int predict(Network *net, float *input, int verbose) {

    // Step 1: normalize input if scaler exists
    // During Iris training, Python's StandardScaler transformed the data:
    // scaled = (raw - mean) / std
    // The weights were trained on scaled data, so we MUST scale here too.
    // If we skip this, we're feeding the network numbers it has never seen.
    // MNIST doesn't need this — handled differently during training.
    float normalized[MAX_FEATURES];  // MAX_FEATURES is the max input size
    if (net->has_scaler) {
        for (int i = 0; i < net->layer_sizes[0]; i++)
            normalized[i] = (input[i] - net->scaler_mean[i]) / net->scaler_std[i];
        input = normalized;
    }

    // Find largest layer to allocate buffers big enough for any layer
    int max_size = 0;
    for (int i = 0; i < net->num_layers; i++)
        if (net->layer_sizes[i] > max_size) max_size = net->layer_sizes[i];

    float *buf_a = malloc(max_size * sizeof(float));
    float *buf_b = malloc(max_size * sizeof(float));

    // Copy input into buf_a to start
    for (int i = 0; i < net->layer_sizes[0]; i++) buf_a[i] = input[i];

    // Loop through all layers generically
    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int in_size  = net->layer_sizes[layer];
        int out_size = net->layer_sizes[layer + 1];
        int is_last  = (layer == net->num_layers - 2);

        // linear transformation: buf_b = W @ buf_a + b
        linear(buf_a, net->weights[layer], net->biases[layer], buf_b, out_size, in_size);

        // Last layer → Softmax (probabilities)
        // All other layers → ReLU (non-linearity)
        if (is_last) softmax(buf_b, out_size);
        else         relu(buf_b, out_size);

        // Swap buffers — output becomes input for next layer
        float *tmp = buf_a; buf_a = buf_b; buf_b = tmp;
    }

    // After last swap, result is in buf_a
    int output_size = net->layer_sizes[net->num_layers - 1];

    // Argmax: find class with highest probability
    // NOTE: different max than inside softmax!
    // softmax's max_val = numerical stability trick
    // this max          = actual prediction (which class won?)
    int best = 0;
    for (int i = 1; i < output_size; i++)
        if (buf_a[i] > buf_a[best]) best = i;

    if (verbose) {
        printf("Probabilities: [");
        for (int i = 0; i < output_size; i++)
            printf("%.3f%s", buf_a[i], i < output_size-1 ? ", " : "");
        printf("]\n");
    }

    free(buf_a);
    free(buf_b);
    return best;
}

void benchmark(Network *net, float *sample_input, int iterations) {
    clock_t start = clock();
    for (int i = 0; i < iterations; i++)
        predict(net, sample_input, 0);
    clock_t end = clock();

    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Iterations:       %d\n", iterations);
    printf("Time:             %.3f seconds\n", seconds);
    printf("Predictions/sec:  %.0f\n", iterations / seconds);
}

#ifndef TINY_ML_INFERENCE_NO_MAIN
int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);

    const char *mode = (argc > 1) ? argv[1] : "iris";
    Network net;

    if (strcmp(mode, "mnist") == 0) {
        printf("=== MNIST (784 -> 128 -> 10) ===\n\n");
        if (!load_weights(&net, "data/mnist_weights.bin")) return 1;
        printf("Weights loaded successfully!\n\n");

        float sample[784] = {0};
        printf("Sample prediction (blank image):\n");
        int pred = predict(&net, sample, 1);
        printf("Predicted digit: %d\n\n", pred);

        printf("--- Benchmark ---\n");
        benchmark(&net, sample, 100000);

    } else {
        printf("=== Iris (4 -> 8 -> 3) ===\n\n");
        if (!load_weights(&net, "data/iris_weights.bin")) return 1;
        printf("Weights loaded successfully!\n\n");

        char *labels[] = {"Setosa", "Versicolor", "Virginica"};
        float flowers[3][4] = {
            {5.1, 3.5, 1.4, 0.2},
            {6.0, 2.9, 4.5, 1.5},
            {6.3, 3.3, 6.0, 2.5},
        };

        for (int i = 0; i < 3; i++) {
            printf("Flower %d: [%.1f, %.1f, %.1f, %.1f]\n",
                   i+1, flowers[i][0], flowers[i][1], flowers[i][2], flowers[i][3]);
            int pred = predict(&net, flowers[i], 1);
            printf("Predicted: %s (class %d)\n\n", labels[pred], pred);
        }

        printf("--- Benchmark ---\n");
        float sample[] = {5.1, 3.5, 1.4, 0.2};
        benchmark(&net, sample, 1000000);
    }

    free_network(&net);
    return 0;
}
#endif
