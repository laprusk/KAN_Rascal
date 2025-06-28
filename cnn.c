#include "cnn.h"
#include <stdlib.h>
#include <math.h>


double Pin[MAX_CHANNELS][HEIGHT + KERNEL_SIZE - 1][WIDTH + KERNEL_SIZE - 1];


void cnn_online_initialize(CNNOnline* model) {

	for (int s = 0; s < model->num_segments; ++s) for (int c = 0; c < model->cnt_conv; ++c) {
		double he_wp = 2 * sqrt(6.0 / (model->channels[s][c] * KERNEL_SIZE * KERNEL_SIZE));
		for (int j = 0; j < model->channels[s][c + 1]; ++j) {		// o—Ích
			model->bias[s][c][j] = 0;
			for (int i = 0; i < model->channels[s][c]; ++i) {			// “ü—Ích
				for (int y = 0; y < KERNEL_SIZE; ++y) for (int x = 0; x < KERNEL_SIZE; ++x) {
					model->weight[s][c][j][i][y][x] = ((double)rand() / (RAND_MAX)-0.5) * he_wp;
				}
			}
		}
	}

}


void conv(double in[MAX_CHANNELS][HEIGHT][WIDTH], double out[MAX_CHANNELS][HEIGHT][WIDTH],
	double cwoi[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE], double cbias[MAX_CHANNELS],
	int ich, int och, int hsize, int wsize) {

	int i, j, h, w, p = KERNEL_SIZE / 2, ph = hsize + KERNEL_SIZE - 1, pw = wsize + KERNEL_SIZE - 1, x, y;

	// 0 padding
	for (i = 0; i < ich; ++i) {
		for (h = 0; h < ph; ++h) for (w = 0; w < pw; ++w) {
			Pin[i][h][w] = 0;
		}
	}
	for (i = 0; i < ich; ++i) {
		for (h = 0; h < hsize; ++h) for (w = 0; w < wsize; ++w) {
			Pin[i][h + p][w + p] = in[i][h][w];
		}
	}

	// conv
	for (j = 0; j < och; ++j) {
		for (h = 0; h < hsize; ++h) for (w = 0; w < wsize; ++w) {
			out[j][h][w] = cbias[j];
			for (i = 0; i < ich; ++i) {
				for (y = 0; y < KERNEL_SIZE; ++y) for (x = 0; x < KERNEL_SIZE; ++x) {
					out[j][h][w] += Pin[i][h + y][w + x] * cwoi[j][i][y][x];
				}
			}
		}
	}

	// relu
	for (j = 0; j < och; ++j) {
		for (h = 0; h < hsize; ++h) for (w = 0; w < wsize; ++w) {
			if (out[j][h][w] < 0) out[j][h][w] = 0;
		}
	}
}


void cnn_online_forward(CNNOnline* model, double* input, double* flattend_output) {

}


void cnn_online_backprop(CNNOnline* model, double* delta_fc) {

}
