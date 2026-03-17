/*
 * CPython C extension for Tiny ML Runtime.
 *
 * Exposes the C inference engine to Python:
 *
 *   import tinymlinference
 *   probs = tinymlinference.predict("data/iris_weights.bin", [5.1, 3.5, 1.4, 0.2])
 *   # → (0, [1.0, 0.0, 0.0])
 *
 * Build: pip install .   (uses setup.py / pyproject.toml)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Inline copy of the inference engine ─────────────────────────── */

#define MAX_LAYERS 10

typedef struct {
    int   num_layers;
    int   layer_sizes[MAX_LAYERS];
    float *weights[MAX_LAYERS];
    float *biases[MAX_LAYERS];
    int   has_scaler;
    /* scaler arrays: one entry per input feature (capped at MAX_LAYERS).
     * Networks with input_size > MAX_LAYERS and a scaler are not supported. */
    float scaler_mean[MAX_LAYERS];
    float scaler_std[MAX_LAYERS];
} Network;

static int load_weights(Network *net, const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f) return 0;

    if (fread(&net->num_layers, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (net->num_layers < 2 || net->num_layers > MAX_LAYERS) { fclose(f); return 0; }
    if (fread(net->layer_sizes, sizeof(int), net->num_layers, f)
            != (size_t)net->num_layers) { fclose(f); return 0; }

    for (int i = 0; i < net->num_layers - 1; i++) {
        int rows = net->layer_sizes[i + 1];
        int cols = net->layer_sizes[i];
        net->weights[i] = malloc(rows * cols * sizeof(float));
        net->biases[i]  = malloc(rows * sizeof(float));
        if (!net->weights[i] || !net->biases[i]) { fclose(f); return 0; }
        if (fread(net->weights[i], sizeof(float), rows * cols, f)
                != (size_t)(rows * cols)) { fclose(f); return 0; }
        if (fread(net->biases[i], sizeof(float), rows, f)
                != (size_t)rows) { fclose(f); return 0; }
    }

    int input_size = net->layer_sizes[0];
    /* scaler arrays are bounded by MAX_LAYERS; skip scaler for large inputs */
    if (input_size <= MAX_LAYERS) {
        size_t n = fread(net->scaler_mean, sizeof(float), input_size, f);
        if (n == (size_t)input_size) {
            if (fread(net->scaler_std, sizeof(float), input_size, f)
                    != (size_t)input_size) { fclose(f); return 0; }
            net->has_scaler = 1;
        } else {
            net->has_scaler = 0;
        }
    } else {
        net->has_scaler = 0;
    }

    fclose(f);
    return 1;
}

static void free_network(Network *net)
{
    for (int i = 0; i < net->num_layers - 1; i++) {
        free(net->weights[i]);
        free(net->biases[i]);
    }
}

static void linear_layer(float *in, float *W, float *b, float *out,
                          int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        out[i] = b[i];
        for (int j = 0; j < cols; j++)
            out[i] += W[i * cols + j] * in[j];
    }
}

static void relu(float *x, int size)
{
    for (int i = 0; i < size; i++)
        if (x[i] < 0) x[i] = 0;
}

static void softmax(float *x, int size)
{
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

/* Returns (class_index, probs_list) or NULL on error.
 * probs_buf must have room for output_size floats. */
static int forward(Network *net, float *input, float *probs_buf)
{
    float *normalized = NULL;
    if (net->has_scaler) {
        int input_size = net->layer_sizes[0];
        normalized = malloc(input_size * sizeof(float));
        if (!normalized) return -1;
        for (int i = 0; i < input_size; i++)
            normalized[i] = (input[i] - net->scaler_mean[i])
                            / net->scaler_std[i];
        input = normalized;
    }

    int max_size = 0;
    for (int i = 0; i < net->num_layers; i++)
        if (net->layer_sizes[i] > max_size) max_size = net->layer_sizes[i];

    float *buf_a = malloc(max_size * sizeof(float));
    float *buf_b = malloc(max_size * sizeof(float));
    if (!buf_a || !buf_b) { free(buf_a); free(buf_b); free(normalized); return -1; }

    for (int i = 0; i < net->layer_sizes[0]; i++) buf_a[i] = input[i];
    free(normalized);
    normalized = NULL;

    for (int layer = 0; layer < net->num_layers - 1; layer++) {
        int in_sz  = net->layer_sizes[layer];
        int out_sz = net->layer_sizes[layer + 1];
        int is_last = (layer == net->num_layers - 2);

        linear_layer(buf_a, net->weights[layer], net->biases[layer],
                     buf_b, out_sz, in_sz);

        if (is_last) softmax(buf_b, out_sz);
        else         relu(buf_b, out_sz);

        float *tmp = buf_a; buf_a = buf_b; buf_b = tmp;
    }

    int output_size = net->layer_sizes[net->num_layers - 1];
    int best = 0;
    for (int i = 1; i < output_size; i++)
        if (buf_a[i] > buf_a[best]) best = i;

    memcpy(probs_buf, buf_a, output_size * sizeof(float));

    free(buf_a);
    free(buf_b);
    return best;
}

/* ── Python binding ───────────────────────────────────────────────── */

/*
 * tinymlinference.predict(weights_file, input_list)
 *   → (class_index: int, probabilities: list[float])
 */
static PyObject *py_predict(PyObject *self, PyObject *args)
{
    const char *weights_file;
    PyObject   *input_list;

    if (!PyArg_ParseTuple(args, "sO!", &weights_file,
                          &PyList_Type, &input_list))
        return NULL;

    Network net;
    memset(&net, 0, sizeof(net));

    if (!load_weights(&net, weights_file)) {
        PyErr_Format(PyExc_IOError,
                     "Failed to load weights from '%s'", weights_file);
        return NULL;
    }

    int input_size = net.layer_sizes[0];
    Py_ssize_t list_len = PyList_Size(input_list);
    if (list_len != input_size) {
        free_network(&net);
        PyErr_Format(PyExc_ValueError,
                     "Input length %zd does not match network input size %d",
                     list_len, input_size);
        return NULL;
    }

    float *input = malloc(input_size * sizeof(float));
    if (!input) { free_network(&net); return PyErr_NoMemory(); }

    for (int i = 0; i < input_size; i++) {
        PyObject *item = PyList_GetItem(input_list, i);
        input[i] = (float)PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            free(input);
            free_network(&net);
            return NULL;
        }
    }

    int output_size = net.layer_sizes[net.num_layers - 1];
    float *probs = malloc(output_size * sizeof(float));
    if (!probs) { free(input); free_network(&net); return PyErr_NoMemory(); }

    int class_idx = forward(&net, input, probs);
    free(input);
    free_network(&net);

    if (class_idx < 0) {
        free(probs);
        return PyErr_NoMemory();
    }

    PyObject *probs_list = PyList_New(output_size);
    if (!probs_list) { free(probs); return NULL; }
    for (int i = 0; i < output_size; i++)
        PyList_SetItem(probs_list, i, PyFloat_FromDouble((double)probs[i]));
    free(probs);

    return Py_BuildValue("(iO)", class_idx, probs_list);
}

/* ── Module definition ────────────────────────────────────────────── */

static PyMethodDef TinyMLMethods[] = {
    {"predict", py_predict, METH_VARARGS,
     "predict(weights_file, input_list) -> (class_index, probabilities)\n\n"
     "Run inference using the C engine.\n\n"
     "Args:\n"
     "    weights_file (str): path to the binary weights file\n"
     "    input_list (list[float]): input feature vector\n\n"
     "Returns:\n"
     "    tuple: (predicted class index, list of class probabilities)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "tinymlinference", NULL, -1, TinyMLMethods
};

PyMODINIT_FUNC PyInit_tinymlinference(void)
{
    return PyModule_Create(&moduledef);
}
