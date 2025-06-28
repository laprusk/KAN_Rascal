#pragma once

#include "dataset.h"

#define MAX_NUM_SEGMENTS 2
#define MAX_COUNT_CONV 2
#define MAX_CHANNELS 16
#define KERNEL_SIZE 3
#define POOL_SIZE 2

typedef struct {
	int num_segments;
	int cnt_conv;
	int channels[MAX_NUM_SEGMENTS][MAX_COUNT_CONV + 1];
	double learning_rate;
	double weight[MAX_NUM_SEGMENTS][MAX_COUNT_CONV][MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
	double bias[MAX_NUM_SEGMENTS][MAX_COUNT_CONV][MAX_CHANNELS];
	double out[MAX_NUM_SEGMENTS][MAX_COUNT_CONV + 1][MAX_CHANNELS][HEIGHT][WIDTH];
	double delta[MAX_NUM_SEGMENTS][MAX_COUNT_CONV + 1][MAX_CHANNELS][HEIGHT][WIDTH];
} CNNOnline;

void cnn_online_initialize(CNNOnline* model);
void cnn_online_forward(CNNOnline* model, double* input, double* flattend_output);
void cnn_online_backprop(CNNOnline* model, double* delta_fc);
