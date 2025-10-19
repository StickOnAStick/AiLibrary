
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

float** load_mnist_images(
    const char* filename, 
    uint32_t *count, 
    uint32_t *rows, 
    uint32_t *cols
){
    FILE *f = fopen(filename, 'rb');
    if (!f) { perror("open imeages"); exit(1); }

    uint32_t magic = read_uint32_be(f);
    if(magic != 2051){
        fprintf(stderr, "Invalid magic number %u for images\n", magic);
        exit(1);
    }

    *count  = read_uint32_be(f);
    *rows   = read_uint32_be(f);
    *cols   = read_uint32_be(f);

    float** images = malloc(*count * sizeof(float *));
    for (uint32_t i = 0; i < *count; ++i){
        images[i] = malloc((*rows) * (*cols) * sizeof(float));
        for(uint32_t j = 0; j < (*rows) * (*cols); ++j){
            unsigned char pixel;
            fread(&pixel, 1, 1, f);
            images[i][j] = pixel / 255.0f; // Normalize
        }
    }

    fclose(f);
    return images;
}

uint8_t* load_mnist_labels(const char *filename, uint32_t *count) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("open imeages"); exit(1); }

    uint32_t magic = read_uint32_be(f);
    if(magic != 2051){
        fprintf(stderr, "Invalid magic number %u for images\n", magic);
        exit(1);
    }

    *count = read_uint32_be(f);
    uint8_t *labels = malloc(*count);
    fread(labels, 1, *count, f);
    fclose(f);
    return labels;
}