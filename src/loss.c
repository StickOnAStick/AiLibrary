#include <loss.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>


float loss_categorical_cross_entropy(float* y_true, float* y_pred, size_t d) {
    static const float eps = 1e-7;
    float sum = 0;
    for(size_t i = 0; i < d; ++i)
        sum += y_true[i] * logf(fmaxf(y_pred[i], eps));
    return -sum;
}