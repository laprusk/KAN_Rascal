#pragma once

#include "dataset.h"
#include "mlp.h"

#define NUM_SEGMENTS 2
#define COUNT_CONV 2
#define MAX_CH 16
#define KERNEL_SIZE 3
#define POOL_SIZE 2

typedef struct {
	int num_segments;
	int cnt_conv;
	int channels[NUM_SEGMENTS][COUNT_CONV + 1];
	double learning_rate;
	double weight[NUM_SEGMENTS][COUNT_CONV][MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE];
	double bias[NUM_SEGMENTS][COUNT_CONV][MAX_CH];
	double out[NUM_SEGMENTS][COUNT_CONV + 1][MAX_CH][HEIGHT][WIDTH];
	double delta[NUM_SEGMENTS][COUNT_CONV + 1][MAX_CH][HEIGHT][WIDTH];
} CNNOnline;

void cnn_init(
	int num_ch[NUM_SEGMENTS][COUNT_CONV + 1],
	double weight[NUM_SEGMENTS][COUNT_CONV][MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE],
	double bias[NUM_SEGMENTS][COUNT_CONV][MAX_CH]
);
void cnn_forward(
	double x_in[DIM],
	int num_ch[NUM_SEGMENTS][COUNT_CONV + 1],
	double weight[NUM_SEGMENTS][COUNT_CONV][MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE],
	double bias[NUM_SEGMENTS][COUNT_CONV][MAX_CH],
	double out[NUM_SEGMENTS][COUNT_CONV + 1][MAX_CH][HEIGHT][WIDTH],
	double flattened_out[MLP_INPUT_DIM]
);
void cnn_backprop(
	double delta_fc[MLP_INPUT_DIM],
	int num_ch[NUM_SEGMENTS][COUNT_CONV + 1],
	double weight[NUM_SEGMENTS][COUNT_CONV][MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE],
	double bias[NUM_SEGMENTS][COUNT_CONV][MAX_CH],
	double out[NUM_SEGMENTS][COUNT_CONV + 1][MAX_CH][HEIGHT][WIDTH],
	double delta[NUM_SEGMENTS][COUNT_CONV + 1][MAX_CH][HEIGHT][WIDTH],
	double dweight[NUM_SEGMENTS][COUNT_CONV][MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE],
	double dbias[NUM_SEGMENTS][COUNT_CONV][MAX_CH]
);

//void cnn_online_initialize(CNNOnline* model);
//void cnn_online_forward(CNNOnline* model, double* input, double* flattend_output);
//void cnn_online_backprop(CNNOnline* model, double* delta_fc);
