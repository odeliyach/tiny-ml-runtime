#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#define DATA_DIR "data"
#ifdef _WIN32
#include <direct.h>
#define TINYML_MKDIR(path) _mkdir(path)
#else
#define TINYML_MKDIR(path) mkdir(path, 0755)
#endif

#include "unity.h"
#include "inference.h"

static void ensure_data_dir(void) {
    int rc = TINYML_MKDIR(DATA_DIR);
    if (rc != 0) {
        TEST_ASSERT_TRUE(errno == EEXIST);
    }
}

static void build_two_layer_identity(Network *net) {
    memset(net, 0, sizeof(*net));
    net->num_layers = 2;
    net->layer_sizes[0] = 2;
    net->layer_sizes[1] = 2;

    size_t weight_count = (size_t)net->layer_sizes[0] * (size_t)net->layer_sizes[1];
    size_t bias_count   = (size_t)net->layer_sizes[1];
    float *W = malloc(weight_count * sizeof(float));
    float *b = malloc(bias_count * sizeof(float));
    TEST_ASSERT_NOT_NULL(W);
    TEST_ASSERT_NOT_NULL(b);
    float init_W[] = {1, 0, 0, 1};
    float init_b[] = {0, 0};
    memcpy(W, init_W, sizeof(init_W));
    memcpy(b, init_b, sizeof(init_b));

    net->weights[0] = W;
    net->biases[0] = b;
}

static void free_two_layer_identity(Network *net) {
    free_network(net);
}

static void test_linear_basic(void) {
    float W[] = {1, 2, 3, 4};
    float b[] = {0.5f, -0.5f};
    float in[] = {1.0f, 1.0f};
    float out[2];

    linear(in, W, b, out, 2, 2);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 3.5f, out[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 6.5f, out[1]);
}

static void test_relu_inplace(void) {
    float x[] = {-1.0f, 0.0f, 5.0f};
    relu(x, 3);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 5.0f, x[2]);
}

static void test_softmax_sum_to_one(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    softmax(x, 3);
    float sum = x[0] + x[1] + x[2];
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, sum);
    TEST_ASSERT_TRUE(x[2] > x[1]);
    TEST_ASSERT_TRUE(x[1] > x[0]);
}

static void test_predict_with_scaler(void) {
    Network net;
    build_two_layer_identity(&net);
    net.has_scaler = 1;
    net.scaler_mean[0] = 5.0f;
    net.scaler_mean[1] = 5.0f;
    net.scaler_std[0] = 1.0f;
    net.scaler_std[1] = 1.0f;

    float input[] = {6.0f, 4.0f};
    int pred = predict(&net, input, 0);

    TEST_ASSERT_EQUAL_INT(0, pred);
    free_two_layer_identity(&net);
}

static void test_predict_three_layer_path(void) {
    Network net;
    memset(&net, 0, sizeof(net));
    net.num_layers = 3;
    net.layer_sizes[0] = 2;
    net.layer_sizes[1] = 3;
    net.layer_sizes[2] = 2;

    size_t W0_count = (size_t)net.layer_sizes[0] * (size_t)net.layer_sizes[1];
    size_t b0_count = (size_t)net.layer_sizes[1];
    size_t W1_count = (size_t)net.layer_sizes[1] * (size_t)net.layer_sizes[2];
    size_t b1_count = (size_t)net.layer_sizes[2];
    float *W0 = malloc(W0_count * sizeof(float));
    float *b0 = malloc(b0_count * sizeof(float));
    float *W1 = malloc(W1_count * sizeof(float));
    float *b1 = malloc(b1_count * sizeof(float));
    TEST_ASSERT_NOT_NULL(W0);
    TEST_ASSERT_NOT_NULL(b0);
    TEST_ASSERT_NOT_NULL(W1);
    TEST_ASSERT_NOT_NULL(b1);

    float init_W0[] = {1, 0, 0, 1, -1, 1};
    float init_b0[] = {0, 0, 0};
    float init_W1[] = {1, 0, 0, 0, 1, 0};
    float init_b1[] = {0, 0};

    memcpy(W0, init_W0, sizeof(init_W0));
    memcpy(b0, init_b0, sizeof(init_b0));
    memcpy(W1, init_W1, sizeof(init_W1));
    memcpy(b1, init_b1, sizeof(init_b1));

    net.weights[0] = W0; net.biases[0] = b0;
    net.weights[1] = W1; net.biases[1] = b1;

    float input[] = {5.0f, 1.0f};
    int pred = predict(&net, input, 0);
    TEST_ASSERT_EQUAL_INT(0, pred);

    free_network(&net);
}

static void write_minimal_weights(const char *path) {
    FILE *f = fopen(path, "wb");
    TEST_ASSERT_NOT_NULL(f);
    int header[] = {2, 2, 2};
    float W[] = {1, 0, 0, 1};
    float b[] = {0, 0};
    fwrite(header, sizeof(int), 3, f);
    fwrite(W, sizeof(float), 4, f);
    fwrite(b, sizeof(float), 2, f);
    fclose(f);
}

static void test_load_weights_success(void) {
    const char *path = DATA_DIR "/test_weights.bin";
    ensure_data_dir();
    write_minimal_weights(path);

    Network net;
    memset(&net, 0, sizeof(net));
    int ok = load_weights(&net, path);
    TEST_ASSERT_EQUAL_INT(1, ok);
    if (ok) {
        TEST_ASSERT_EQUAL_INT(2, net.num_layers);
        TEST_ASSERT_EQUAL_INT(2, net.layer_sizes[0]);
        TEST_ASSERT_EQUAL_INT(2, net.layer_sizes[1]);
        free_network(&net);
    }
}

static void test_load_weights_missing_file(void) {
    Network net;
    memset(&net, 0, sizeof(net));
    int ok = load_weights(&net, DATA_DIR "/does_not_exist.bin");
    TEST_ASSERT_EQUAL_INT(0, ok);
    TEST_ASSERT_EQUAL_INT(0, net.num_layers);
}

static void test_load_weights_truncated(void) {
    const char *path = DATA_DIR "/truncated.bin";
    ensure_data_dir();
    FILE *f = fopen(path, "wb");
    TEST_ASSERT_NOT_NULL(f);
    int header[] = {3, 2, 2, 2};
    fwrite(header, sizeof(int), 4, f);
    fclose(f);

    Network net;
    memset(&net, 0, sizeof(net));
    int ok = load_weights(&net, path);
    TEST_ASSERT_EQUAL_INT(0, ok);
    TEST_ASSERT_EQUAL_INT(0, net.num_layers);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_linear_basic);
    RUN_TEST(test_relu_inplace);
    RUN_TEST(test_softmax_sum_to_one);
    RUN_TEST(test_predict_with_scaler);
    RUN_TEST(test_predict_three_layer_path);
    RUN_TEST(test_load_weights_success);
    RUN_TEST(test_load_weights_missing_file);
    RUN_TEST(test_load_weights_truncated);
    return UNITY_END();
}
