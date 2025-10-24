#include <stdio.h>
#include <layer.h>
#include <memory.h>

#include "activations.h"
#include "mnist.h"
#include "loss.h"

#define D_I   28*28
#define D_INT 128
#define D_O   10
#define BATCH_SIZE 10

int main(int argc, char** argv){
    
    printf("Starting perceptron");
    
    uint32_t count, rows, cols;
    float **images  = mnist_load_images("./data/train-images-idx3-ubyte", &count, &rows, &cols);
    uint8_t *labels = mnist_load_labels("./data/train-labels-idx1-ubyte", &count);

    layer_t *layer_i    = layer_init(D_I, D_INT);
    layer_t *layer_o    = layer_init(D_INT, D_O);

    float total_loss = 0;

    for(size_t i = 0; i < count / BATCH_SIZE; i++){
        
        float* batch_input = malloc(BATCH_SIZE * D_I * sizeof(float));
        float* batch_labels = malloc(BATCH_SIZE * sizeof(float));
        
        for(size_t j = 0; j < BATCH_SIZE; j++){
            batch_labels[j] = labels[i * BATCH_SIZE + j];
            memcpy(&batch_input[j * D_I], images[i * BATCH_SIZE + j], D_I * sizeof(float));
        }

        // === Forward ===
        float* h = layer_forward(layer_i, batch_input, BATCH_SIZE);
        relu_vec(h, D_INT);
        float* out = layer_forward(layer_o, h, BATCH_SIZE);
        softmax(out, D_O);

        // === Loss ===
        float loss = loss_categorical_cross_entropy(labels, out, BATCH_SIZE);
        total_loss += loss;
        printf("Epoch %u loss: %.4f\tTotal: %.4f", loss, total_loss);

        // === Backward ===
        float* grad_l1 = layer_backward(layer_o, out, BATCH_SIZE);
        

    }

    // Forward
    float* h    = layer_forward(layer_i, images[0], 1);
    relu_vec(h, D_INT);
    float* out  = layer_forward(layer_o, h, 1);
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