#include <stdio.h>
#include <layer.h>
#include "activations.h"
#include "mnist.h"
#include <stddef.h>

#define D_I  28*28
#define D_INT 128
#define D_O  10

int main(int argc, char** argv){
    
    printf("Starting perceptron");
    
    uint32_t count, rows, cols;
    float **images = mnist_load_images("./data/train-images-idx3-ubyte", &count, &rows, &cols);
    uint8_t *labels = mnist_load_labels("./data/train-labels-idx1-ubyte", &count);

    layer_t *layer_i    = layer_init(D_I, D_INT);
    layer_t *layer_o  = layer_init(D_INT, D_O);

    // Forward
    float* h = layer_forward(layer_i, images[0], 1);
    relu_vec(h, D_INT);
    float* out = layer_forward(layer_o, h, 1);
    softmax(out, 10);

    size_t pred = argxmax(out, D_O);
    printf("Predicted label: %zu, true label: %u\n", pred, labels[0]);

    free(h);
    free(out);
    layer_free(layer_i);
    layer_free(layer_o);

    
    return 0;
}

void arg_parse(int argc, char** argv){
    /*
        Args:
            

    */

}