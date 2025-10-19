#pragma once
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef float (*active_f_fn)(float);
typedef void (*active_fp_fn)(float*, size_t);

static float relu(float x);
static float sigmoid(float x);
static float tanh_act(float x);

static void relu_vec(float* x, size_t d);
static void sigmoid_vec(float* x, size_t d);
static void softmax(float* x, size_t d);

size_t argxmax(const float* x, size_t d);

#endif // Activation