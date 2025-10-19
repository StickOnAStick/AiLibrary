#pragma once
#ifndef LOAD_MNIST_H
#define LOAD_MNIST_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

static uint32_t read_uint32_be(FILE *f) {
    uint8_t b[4];
    fread(b, 1, 4, f);
    return (b[0] << 24 | (b[1] << 16) | (b[2] << 8) | b[3]);
}

float** load_mnist_images(const char *filename, uint32_t *count, uint32_t *rows, uint32_t *cols);
uint8_t* load_mnist_labels(const char* filename, uint32_t *count);

#endif