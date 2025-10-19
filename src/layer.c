#include <layer.h>
#include <math.h>


layer_t* layer_init(size_t d_i, size_t d_o) {
    layer_t* res = (layer_t*)malloc(sizeof(layer_t));
    if(!res) return NULL;

    res->in_features = d_i;
    res->out_features = d_o;

    float* w =  (float*)malloc(sizeof(float) * (d_o * d_i));
    float* b =  (float*)malloc(sizeof(float) * d_o);
    float* gw = (float*)calloc(d_o * d_i, sizeof(float));
    float* gb = (float*)calloc(d_o, sizeof(float));
    float* ic = NULL;

    if ( !w || !b || !gw || !gb ) {
        free(w);
        free(b);
        free(gw);
        free(gb);
        return NULL;
    }

    res->weight = w;
    res->bias   = b;
    res->grad_weight =  gw;
    res->grad_bias = gb;
    res->input_cache = ic;

    float limit = sqrtf(6.0f / (d_i + d_o));
    for(size_t i = 0; i < d_o*d_i; i++)
        res->weight[i] = rand_uniform(-limit, limit);
    for(size_t i = 0; i < d_o; i++)
        res->bias[i] = 0.0f;

    return res;
};

void layer_free(layer_t* l){
    if(!l) return;
    free(l->weight);
    free(l->bias);
    free(l->grad_weight);
    free(l->grad_bias);
    free(l->input_cache);
    free(l);
}

float* layer_forward(layer_t* l, const float* input, size_t batch_size) {

    size_t in = l->in_features, out = l->out_features;
    float* output = calloc(batch_size * out, sizeof(float));

    free(l->input_cache);
    l->input_cache = malloc(sizeof(float) * batch_size * in);
    memcpy(l->input_cache, input, sizeof(float) * batch_size * in);

    for(size_t b = 0; b < batch_size; ++b){
        for(size_t o = 0; o < out; ++o){
            float sum = l->bias[o]; // Toss in early.
            for(size_t i = 0; i < in; ++i)
                sum += l->weight[o * in + i] * input[b  * in + i];
            output[b * out + o] = sum;
        }
    }

    return output;
}

float* layer_backward(layer_t* l, const float* grad_output, uint32_t batch_size) {
    size_t in = l->in_features, out = l->out_features;
    float* grad_input = calloc(batch_size * in, sizeof(float));

    memset(l->grad_weight, 0, sizeof(float) * in * out);
    memset(l->grad_bias, 0, sizeof(float) * out);

    for(size_t b = 0; b < batch_size; ++b){
        for(size_t o = 0; o < out; ++ o){
            float go = grad_output[b * out + o];
            l->grad_bias[o] += go; // dL/db = dL/dy * dy/db = g_o * 1 (activation loss * L == dL/dy)
            for(size_t i = 0; i < in; ++i){
                l->grad_weight[o * in + i] += go * l->input_cache[b * in + 1]; // dL/dy = dL/dy *dy/dw = g_o * x_i!
                grad_input[b * in + i] += l->weight[o * in + i] * go;
            }
        }
    }

    // averaget he gradients
    for(size_t o = 0; o < out; ++o){
        l->grad_bias[o] /= batch_size;
        for(size_t i = 0; i < in; i++)
            l->grad_weight[o * in + i] /= batch_size;
    }

    return grad_output;
}

void layer_update(layer_t* l, float lr){
    size_t in = l->in_features, out = l->out_features;
    for(size_t o = 0; o < out; ++o){
        for(size_t i = 0; i < in; ++i)
            l->weight[o * in + i] -= lr * l->grad_weight[o * in + i];
        l->bias[o] -= lr * l->grad_bias[o];
    }
}