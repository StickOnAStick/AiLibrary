#include <loss.h>
#include <stdint.h>
#include <math.h>

float loss_categorical_cross_entropy(float* y_true, float* y_pred, size_t d) {
    float sum = 0;
    for(size_t i = 0; i < d; ++i)
        sum += y_true[i] * logf(y_pred[i]);
    return -sum;
}