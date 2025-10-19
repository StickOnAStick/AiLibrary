#pragma once
#ifndef LAYER_H
#define LAYER_H

#include <stdint.h>
#include <stdlib.h> // rand


typedef struct {
    size_t in_features;
    size_t out_features;

    float* weight; // [ out_featurs, in_features ]
    float* bias;  // [ out_features ]
    float* grad_weight; // Same as weight
    float* grad_bias;   // Same as bias

    float* input_cache; // Store for backprop

} layer_t;

static inline float rand_uniform(float min, float max) {
    return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

layer_t*    layer_init(size_t d_i, size_t d_o);
void        layer_free(layer_t* l);
float*      layer_forward(layer_t* l, const float* input, size_t batch_size);
float*      layer_backward(layer_t* l, const float* grad_output, uint32_t batch_size);
void        layer_update(layer_t* l, float lr);

#endif // Layer.h