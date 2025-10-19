#include <stdint.h>
#include <math.h>



static float relu(float x)      { return x > 0 ? x : 0; }
static float sigmoid(float x)   { return 1.0f / (1.0f + expf(-x)); }
static float tanh_act(float x)  { return tanhf(x); }

static void relu_vec(float* x, size_t d) {
    for(size_t i = 0; i < d; i++){
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}
static void sigmoid_vec(float* x, size_t d) {
    for (size_t i = 0; i < d; i++) {
        if (x[i] >= 0) {
            float z = expf(-x[i]);
            x[i] = 1.0f / (1.0f + z);
        } else {
            float z = expf(x[i]);
            x[i] = z / (1.0f + z);
        }
    }
}

static void softmax(float* x, size_t d) {
    float max = x[0];
    for(size_t i = 1; i < d; i++)
        if(x[i] > max) max = x[i];
    
    float sum = 0.0f;
    for(size_t i = 0; i < d; i++){
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for(size_t i = 0; i < d; i++)
        x[i] /= sum;
}

size_t argxmax(const float* x, size_t d) {
    size_t idx = 0;
    float maxv = x[0];
    for(size_t i = 1; i < d; i++)
        if (x[i] > maxv) { maxv = x[i]; idx = i; }
    return idx;
}