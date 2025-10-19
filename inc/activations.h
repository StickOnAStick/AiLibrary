#pragma once
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef float (*active_f_fn)(float);
typedef void (*active_fp_fn)(float*, size_t);

float relu(float x);
float sigmoid(float x);
float tanh_act(float x);
void relu_vec(float* x, size_t d);
void sigmoid_vec(float* x, size_t d);
void softmax(float* x, size_t d);

size_t argxmax(const float* x, size_t d);

#endif // Activation