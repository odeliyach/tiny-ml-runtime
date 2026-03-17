# Interview Guide: Tiny ML Runtime

## 30-Second Elevator Pitch

"I built a neural network inference engine in pure C to understand what frameworks like PyTorch do under the hood. The project demonstrates low-level systems programming, memory management, and Python-C integration. On tiny networks, my C implementation is 258x faster than PyTorch because it eliminates Python dispatch overhead, but on larger networks PyTorch wins 5x due to BLAS optimizations—this crossover point reveals exactly where framework overhead matters versus raw computational throughput."

## Top 5 Technical Questions & Answers

### 1. **Q: Why is your C code 258x faster than PyTorch on Iris but 5x slower on MNIST?**

**A:** It's all about the overhead-to-computation ratio:

- **Iris (4 inputs)**: The actual math (4→8→3) takes ~50 FLOPs. PyTorch spends more time in Python interpreter overhead and C++ dispatch than doing the computation itself. My C code has zero overhead—direct function calls—so it dominates.

- **MNIST (784 inputs)**: The first layer alone (784→128) requires 100,000 multiply-add operations. PyTorch calls into optimized BLAS libraries (OpenBLAS, Accelerate) with SIMD vectorization. My code uses naive nested loops—no SIMD, no cache tiling—so PyTorch's raw FLOP throughput wins.

**Key insight**: This crossover demonstrates the tradeoff between framework convenience and performance. For tiny embedded models, C wins. For production-scale ML, frameworks win.

### 2. **Q: How do you handle memory management in C, especially for dynamically-sized networks?**

**A:** Three strategies:

1. **Dynamic allocation**: Weights/biases are `malloc`'d at runtime because network size is unknown at compile time. I free them in `free_network()` to prevent leaks.

2. **Ping-pong buffers**: Instead of allocating a new buffer per layer, I allocate two buffers sized for the largest layer and ping-pong between them: layer 0 writes to buf_b, layer 1 reads buf_b and writes to buf_a, etc. This reduces allocations from O(n) to O(1).

3. **Stack vs heap tradeoff**: Small arrays (e.g., normalized input up to 784 floats) go on the stack for speed. Large arrays (weights matrices) go on the heap to avoid stack overflow.

**Interview tip**: Mention valgrind for leak detection in production.

### 3. **Q: Explain the "softmax numerical stability trick" in your code.**

**A:** Softmax is `exp(x_i) / sum(exp(x_j))`. If any `x_i` is large (e.g., 1000), `exp(1000)` = infinity → NaN.

**The fix** (inference.c:110-112):
```c
float max_val = x[0];
for (int i = 1; i < size; i++)
    if (x[i] > max_val) max_val = x[i];

for (int i = 0; i < size; i++)
    x[i] = expf(x[i] - max_val);  // ← subtract max first
```

**Why it works**: `exp(x - max) / sum(exp(x_j - max))` = `exp(x) / sum(exp(x_j))` mathematically (constants cancel), but now the largest exponent is 0, so `exp(0)` = 1—no overflow.

**This is a standard production pattern** used in TensorFlow, PyTorch, etc.

### 4. **Q: Why do you need ReLU? Why not just stack linear layers?**

**A:** Without non-linearity, stacking layers is pointless:
```
y = W2 @ (W1 @ x) = (W2 @ W1) @ x = W_combined @ x
```
Multiple layers collapse into a single linear transformation. ReLU breaks linearity:
```c
void relu(float *x, int size) {
    for (int i = 0; i < size; i++)
        if (x[i] < 0) x[i] = 0;
}
```

This introduces non-linear decision boundaries, allowing the network to learn complex patterns (e.g., XOR, which is impossible with pure linear layers).

**Interview tip**: Mention other activations (sigmoid, tanh) and why ReLU is preferred (no vanishing gradients).

### 5. **Q: How did you expose your C code to Python? Walk me through the CPython C-API.**

**A:** I wrote a CPython extension module (inference_module.c):

1. **Define the function**:
```c
static PyObject *py_predict(PyObject *self, PyObject *args) {
    // Parse Python arguments into C types
    const char *weights_file;
    PyObject *input_list;
    PyArg_ParseTuple(args, "sO!", &weights_file, &PyList_Type, &input_list);

    // Call C inference engine
    int class_idx = forward(&net, input, probs);

    // Convert C results back to Python
    return Py_BuildValue("(iO)", class_idx, probs_list);
}
```

2. **Register the module**:
```c
static PyMethodDef TinyMLMethods[] = {
    {"predict", py_predict, METH_VARARGS, "...docstring..."},
    {NULL, NULL, 0, NULL}
};
```

3. **Build with setuptools**: `pip install .` compiles the C code and links it as a Python extension.

**Key challenges**:
- **Reference counting**: Must `Py_INCREF`/`Py_DECREF` properly to avoid memory leaks
- **Error handling**: Use `PyErr_Format()` to raise Python exceptions from C
- **Type conversion**: `PyFloat_AsDouble()`, `PyList_GetItem()`, etc.

**Interview tip**: Mention alternatives (ctypes, CFFI, pybind11) and why CPython C-API gives the most control.

## Core Skills Demonstrated

### 1. **Memory Management & Systems Programming**
- Manual malloc/free with leak prevention
- Buffer reuse patterns (ping-pong buffers)
- Understanding stack vs heap tradeoffs
- Pointer arithmetic for matrix indexing in flat arrays

### 2. **Polyglot Programming (Python ↔ C Integration)**
- CPython C-API for building native extensions
- Cross-language type conversion and error handling
- Build system integration (setup.py, pyproject.toml)
- Understanding when to drop down to C for performance

### 3. **Performance Engineering & Profiling**
- Benchmarking methodology (1M iterations, single-threaded)
- Identifying overhead vs computation bottlenecks
- Understanding SIMD/BLAS optimization (even if not implementing it)
- Recognizing crossover points between implementations

## Technical Deep-Dive Questions (If Interviewer Asks)

### Q: How would you optimize your C code further?
**A:**
1. **SIMD vectorization**: Use SSE/AVX intrinsics for matrix multiplication (4-8 floats per instruction)
2. **Loop tiling**: Block matrix multiplication to fit in L1 cache
3. **Compiler hints**: `__restrict__` pointers, `-march=native`
4. **Quantization**: Use int8 instead of float32 (4x memory reduction, faster on modern CPUs)

### Q: What if the weight file is corrupted?
**A:** Current code has minimal validation. Production improvements:
1. Add a magic number header (e.g., 0x544D4C52 = "TMLR")
2. Store a checksum (CRC32 or SHA256)
3. Validate array bounds (num_layers < MAX_LAYERS, no negative sizes)
4. Check fread return values (already done in inference_module.c)

### Q: How would you make this thread-safe?
**A:** Current code is thread-safe for *read-only inference* (no global state), but **not** for concurrent writes. For multi-threading:
1. Each thread gets its own `Network` struct (no shared mutable state)
2. Weights can be shared read-only across threads
3. Use thread-local buffers for forward pass (buf_a, buf_b)

### Q: Could you extend this to support backpropagation?
**A:** Yes, but requires:
1. Storing activations during forward pass (needed for gradient computation)
2. Implementing backward pass for each layer (chain rule)
3. Gradient descent update: `W -= learning_rate * dL/dW`
4. 3-5x more code complexity

**Better answer**: For training, use PyTorch/JAX (autograd is hard to get right). Use C for inference only.

## Key Achievements for Resume/LinkedIn

- **Built production-grade neural network inference engine in pure C** (zero dependencies)
- **Achieved 258x speedup over PyTorch** on small networks through manual memory optimization
- **Developed CPython C extension** exposing C functions to Python with proper error handling
- **Implemented numerical stability techniques** (softmax max-subtraction) used in production ML frameworks
- **Created comprehensive benchmark suite** demonstrating understanding of performance tradeoffs

## Projects That Pair Well With This

If interviewer asks "What else have you built?", complement this with:
- A web backend (shows you understand networked systems, not just algorithms)
- A distributed system project (shows scalability thinking)
- A GPU/CUDA project (natural extension—"I'd implement SIMD next")
- Any DevOps/CI project (shows you understand the full software lifecycle)

## Red Flags to Avoid

**Don't say**:
- "I just followed a tutorial" (shows lack of depth)
- "C is always faster than Python" (wrong—see MNIST results)
- "This is production-ready" (it's not—missing input validation, error recovery, etc.)

**Do say**:
- "I built this to understand ML frameworks from first principles"
- "The crossover point taught me when frameworks justify their overhead"
- "In production, I'd add [specific validation/logging/monitoring]"

## Closing Statement

"This project gave me hands-on experience with systems programming, performance optimization, and the design decisions behind ML frameworks. I now have a much deeper appreciation for what PyTorch does under the hood—and when to use C vs. when to trust the framework. For my next project, I'm interested in [related area: GPU programming / distributed systems / embedded ML]."
