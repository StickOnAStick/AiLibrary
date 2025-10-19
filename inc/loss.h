#pragma once
#ifndef LOSS_H
#define LOSS_H

#include <math.h>
#include <stdint.h>
#include <stddef.h>


float loss_categorical_cross_entropy(float* y_true, float* y_pred, size_t d);

#endif