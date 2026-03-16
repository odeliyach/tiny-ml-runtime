/*
 * Unit tests for the Tiny ML Runtime inference engine.
 *
 * Tests core functions (linear, relu, softmax, predict) with known inputs
 * and expected outputs. No external weight files required — tests construct
 * small networks in-memory.
 *
 * Build:  gcc -Wall -Wextra -O2 -o test_inference test_inference.c -lm
 * Run:    ./test_inference
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ── Copy of core types and functions from inference.c ────────────── */

#define MAX_LAYERS 10

typedef struct {
    int   num_layers;
    int   layer_sizes[MAX_LAYERS];
    float *weights[MAX_LAYERS];
    float *biases[MAX_LAYERS];
    int   has_scaler;
    float scaler_mean[MAX_LAYERS];
    float scaler_std[MAX_LAYERS];
} Network;

void free_network(Network *net) {
    for (int i = 0; i < net->num_layers - 1; i++) {
        free(net->weights[i]);
        free(net->biases[i]);
    }
}

void linear(float *in, float *W, float *b, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        out[i] = b[i];
        for (int j = 0; j < cols; j++)
            out[i] += W[i * cols + j] * in[j];
    }
}

void relu(float *x, int size) {
    for (int i = 0; i < size; i++)
        if (x[i] < 0) x[i] = 0;
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
    float normalized[784];  /* 784 = max input size (28×28 MNIST) */
    if (net->has_scaler) {
        for (int i = 0; i < net->layer_sizes[0]; i++)
            normalized[i] = (input[i] - net->scaler_mean[i]) / net->scaler_std[i];
        input = normalized;
    }

    int max_size = 0;
    for (int i = 0; i < net->num_layers; i++)
        if (net->layer_sizes[i] > max_size) max_size = net->layer_sizes[i];

    float *buf_a = malloc(max_size * sizeof(float));
    float *buf_b = malloc(max_size * sizeof(float));

    for (int i = 0; i < net->layer_sizes[0]; i++) buf_a[i] = input[i];

    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int in_size  = net->layer_sizes[layer];
        int out_size = net->layer_sizes[layer + 1];
        int is_last  = (layer == net->num_layers - 2);

        linear(buf_a, net->weights[layer], net->biases[layer], buf_b, out_size, in_size);

        if (is_last) softmax(buf_b, out_size);
        else         relu(buf_b, out_size);

        float *tmp = buf_a; buf_a = buf_b; buf_b = tmp;
    }

    int output_size = net->layer_sizes[net->num_layers - 1];
    int best = 0;
    for (int i = 1; i < output_size; i++)
        if (buf_a[i] > buf_a[best]) best = i;

    if (verbose) {
        printf("  Probabilities: [");
        for (int i = 0; i < output_size; i++)
            printf("%.3f%s", buf_a[i], i < output_size - 1 ? ", " : "");
        printf("]\n");
    }

    free(buf_a);
    free(buf_b);
    return best;
}

/* ── Test helpers ─────────────────────────────────────────────────── */

static int tests_run    = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg)                                              \
    do {                                                               \
        tests_run++;                                                   \
        if (cond) { tests_passed++; }                                  \
        else {                                                         \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__);           \
        }                                                              \
    } while (0)

#define ASSERT_NEAR(a, b, eps, msg)                                    \
    ASSERT(fabsf((a) - (b)) < (eps), msg)

/* ── Tests ────────────────────────────────────────────────────────── */

void test_relu(void) {
    printf("test_relu\n");

    float x[] = {-3.0f, -1.0f, 0.0f, 1.0f, 5.0f};
    relu(x, 5);

    ASSERT_NEAR(x[0], 0.0f, 1e-6, "relu(-3) == 0");
    ASSERT_NEAR(x[1], 0.0f, 1e-6, "relu(-1) == 0");
    ASSERT_NEAR(x[2], 0.0f, 1e-6, "relu(0) == 0");
    ASSERT_NEAR(x[3], 1.0f, 1e-6, "relu(1) == 1");
    ASSERT_NEAR(x[4], 5.0f, 1e-6, "relu(5) == 5");
}

void test_softmax_basic(void) {
    printf("test_softmax_basic\n");

    float x[] = {1.0f, 2.0f, 3.0f};
    softmax(x, 3);

    float sum = x[0] + x[1] + x[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5, "softmax sums to 1");
    ASSERT(x[2] > x[1] && x[1] > x[0], "softmax preserves order");
}

void test_softmax_numerical_stability(void) {
    printf("test_softmax_numerical_stability\n");

    float x[] = {1000.0f, 1001.0f, 1002.0f};
    softmax(x, 3);

    float sum = x[0] + x[1] + x[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5, "softmax stable with large inputs");
    ASSERT(x[0] > 0.0f && x[1] > 0.0f && x[2] > 0.0f,
           "no NaN/Inf from large values");
}

void test_softmax_uniform(void) {
    printf("test_softmax_uniform\n");

    float x[] = {5.0f, 5.0f, 5.0f};
    softmax(x, 3);

    ASSERT_NEAR(x[0], 1.0f / 3.0f, 1e-5, "uniform input → equal probs [0]");
    ASSERT_NEAR(x[1], 1.0f / 3.0f, 1e-5, "uniform input → equal probs [1]");
    ASSERT_NEAR(x[2], 1.0f / 3.0f, 1e-5, "uniform input → equal probs [2]");
}

void test_linear(void) {
    printf("test_linear\n");

    /* W = [[1,2],[3,4]], b = [0.5, -0.5], in = [1, 1]
     * out[0] = 0.5 + 1*1 + 2*1 = 3.5
     * out[1] = -0.5 + 3*1 + 4*1 = 6.5 */
    float W[] = {1, 2, 3, 4};
    float b[] = {0.5f, -0.5f};
    float in[] = {1.0f, 1.0f};
    float out[2];

    linear(in, W, b, out, 2, 2);

    ASSERT_NEAR(out[0], 3.5f, 1e-5, "linear out[0] == 3.5");
    ASSERT_NEAR(out[1], 6.5f, 1e-5, "linear out[1] == 6.5");
}

void test_linear_nonsquare(void) {
    printf("test_linear_nonsquare\n");

    /* W = [[1,0,2]], b = [1], in = [3,5,7]
     * out[0] = 1 + 1*3 + 0*5 + 2*7 = 18 */
    float W[] = {1, 0, 2};
    float b[] = {1.0f};
    float in[] = {3.0f, 5.0f, 7.0f};
    float out[1];

    linear(in, W, b, out, 1, 3);

    ASSERT_NEAR(out[0], 18.0f, 1e-5, "linear 3→1: out[0] == 18");
}

void test_predict_identity_network(void) {
    printf("test_predict_identity_network\n");

    /* 2-layer network (2 → 2) with identity weights and zero biases.
     * Input goes through: linear → softmax, so output = softmax(input).
     * Feeding [10, 0] → softmax([10,0]) → class 0. */
    Network net;
    memset(&net, 0, sizeof(net));
    net.num_layers = 2;
    net.layer_sizes[0] = 2;
    net.layer_sizes[1] = 2;
    net.has_scaler = 0;

    /* Identity matrix: [[1,0],[0,1]] */
    float W[] = {1, 0, 0, 1};
    float b[] = {0, 0};
    net.weights[0] = W;
    net.biases[0]  = b;

    float input_a[] = {10.0f, 0.0f};
    int pred_a = predict(&net, input_a, 0);
    ASSERT(pred_a == 0, "identity net: [10,0] → class 0");

    float input_b[] = {0.0f, 10.0f};
    int pred_b = predict(&net, input_b, 0);
    ASSERT(pred_b == 1, "identity net: [0,10] → class 1");

    /* Don't free — weights are stack-allocated */
}

void test_predict_three_layer(void) {
    printf("test_predict_three_layer\n");

    /* 3-layer network: 2 → 3 → 2
     * Layer 0→1: W0 (3×2), b0 (3)  + ReLU
     * Layer 1→2: W1 (2×3), b1 (2)  + Softmax
     *
     * W0 = [[1,0],[0,1],[-1,1]], b0 = [0,0,0]
     * After linear with input [5,1]:
     *   h = [5, 1, -4] → ReLU → [5, 1, 0]
     *
     * W1 = [[1,0,0],[0,1,0]], b1 = [0,0]
     * After linear: out = [5, 1] → softmax → class 0
     */
    Network net;
    memset(&net, 0, sizeof(net));
    net.num_layers = 3;
    net.layer_sizes[0] = 2;
    net.layer_sizes[1] = 3;
    net.layer_sizes[2] = 2;
    net.has_scaler = 0;

    float W0[] = {1, 0,  0, 1,  -1, 1};
    float b0[] = {0, 0, 0};
    float W1[] = {1, 0, 0,  0, 1, 0};
    float b1[] = {0, 0};

    net.weights[0] = W0;  net.biases[0] = b0;
    net.weights[1] = W1;  net.biases[1] = b1;

    float input[] = {5.0f, 1.0f};
    int pred = predict(&net, input, 0);
    ASSERT(pred == 0, "3-layer net: [5,1] → class 0");
}

void test_predict_with_scaler(void) {
    printf("test_predict_with_scaler\n");

    /* 2-layer network (2 → 2) with identity weights.
     * Scaler: mean=[5,5], std=[1,1]
     * Input [6, 4] → scaled to [1, -1].
     * softmax([1,-1]) → class 0 */
    Network net;
    memset(&net, 0, sizeof(net));
    net.num_layers = 2;
    net.layer_sizes[0] = 2;
    net.layer_sizes[1] = 2;
    net.has_scaler = 1;
    net.scaler_mean[0] = 5.0f;
    net.scaler_mean[1] = 5.0f;
    net.scaler_std[0] = 1.0f;
    net.scaler_std[1] = 1.0f;

    float W[] = {1, 0, 0, 1};
    float b[] = {0, 0};
    net.weights[0] = W;
    net.biases[0]  = b;

    float input[] = {6.0f, 4.0f};
    int pred = predict(&net, input, 0);
    ASSERT(pred == 0, "scaler net: [6,4] → scaled [1,-1] → class 0");
}

int load_weights(Network *net, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return 0;

    if (fread(&net->num_layers, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (fread(net->layer_sizes, sizeof(int), net->num_layers, f)
            != (size_t)net->num_layers) { fclose(f); return 0; }

    for (int i = 0; i < net->num_layers - 1; i++) {
        int rows = net->layer_sizes[i + 1];
        int cols = net->layer_sizes[i];
        net->weights[i] = malloc(rows * cols * sizeof(float));
        net->biases[i]  = malloc(rows * sizeof(float));
        if (fread(net->weights[i], sizeof(float), rows * cols, f)
                != (size_t)(rows * cols)) { fclose(f); return 0; }
        if (fread(net->biases[i], sizeof(float), rows, f)
                != (size_t)rows) { fclose(f); return 0; }
    }

    int input_size = net->layer_sizes[0];
    size_t n = fread(net->scaler_mean, sizeof(float), input_size, f);
    if (n == (size_t)input_size) {
        if (fread(net->scaler_std, sizeof(float), input_size, f)
                != (size_t)input_size) { fclose(f); return 0; }
        net->has_scaler = 1;
    } else {
        net->has_scaler = 0;
    }

    fclose(f);
    return 1;
}

void test_load_and_predict_iris(void) {
    printf("test_load_and_predict_iris\n");

    /* Try loading iris weights — skip if file not present */
    FILE *f = fopen("iris_weights.bin", "rb");
    if (!f) {
        printf("  SKIP: iris_weights.bin not found\n");
        return;
    }
    fclose(f);

    Network net;
    int ok = load_weights(&net, "iris_weights.bin");
    ASSERT(ok == 1, "load iris_weights.bin succeeds");

    if (ok) {
        ASSERT(net.num_layers == 3, "iris has 3 layers");
        ASSERT(net.layer_sizes[0] == 4,  "iris input size == 4");
        ASSERT(net.layer_sizes[1] == 8,  "iris hidden size == 8");
        ASSERT(net.layer_sizes[2] == 3,  "iris output size == 3");

        /* Setosa sample — should predict class 0 */
        float setosa[] = {5.1f, 3.5f, 1.4f, 0.2f};
        int pred = predict(&net, setosa, 0);
        ASSERT(pred == 0, "iris: setosa → class 0");

        free_network(&net);
    }
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Tiny ML Runtime — Unit Tests ===\n\n");

    test_relu();
    test_softmax_basic();
    test_softmax_numerical_stability();
    test_softmax_uniform();
    test_linear();
    test_linear_nonsquare();
    test_predict_identity_network();
    test_predict_three_layer();
    test_predict_with_scaler();
    test_load_and_predict_iris();

    printf("\n%d / %d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
