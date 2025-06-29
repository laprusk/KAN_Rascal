#include "cnn.h"
#include <stdlib.h>
#include <math.h>


double pad_in[MAX_CH][HEIGHT + KERNEL_SIZE - 1][WIDTH + KERNEL_SIZE - 1];
double pad_delta[MAX_CH][HEIGHT + KERNEL_SIZE - 1][WIDTH + KERNEL_SIZE - 1];


void cnn_init(
	int num_ch[NUM_SEGMENTS][COUNT_CONV + 1],
	double weight[NUM_SEGMENTS][COUNT_CONV][MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE],
	double bias[NUM_SEGMENTS][COUNT_CONV][MAX_CH]
) {

	for (int s = 0; s < NUM_SEGMENTS; ++s) for (int c = 0; c < COUNT_CONV; ++c) {
		double he_wp = 2 * sqrt(6.0 / (num_ch[s][c] * KERNEL_SIZE * KERNEL_SIZE));
		for (int j = 0; j < num_ch[s][c + 1]; ++j) {		// 出力ch
			bias[s][c][j] = 0;
			for (int i = 0; i < num_ch[s][c]; ++i) {			// 入力ch
				for (int y = 0; y < KERNEL_SIZE; ++y) for (int x = 0; x < KERNEL_SIZE; ++x) {
					weight[s][c][j][i][y][x] = ((double)rand() / (RAND_MAX)-0.5) * he_wp;
				}
			}
		}
	}

}


void conv(
	double in[MAX_CH][HEIGHT][WIDTH],
	double out[MAX_CH][HEIGHT][WIDTH],
	double weight[MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE],
	double bias[MAX_CH],
	int ich,
	int och,
	int height,
	int width
) {

	int p = KERNEL_SIZE / 2;
	int ph = height + KERNEL_SIZE - 1;
	int pw = width + KERNEL_SIZE - 1;

	// 0 padding
	for (int i = 0; i < ich; ++i) {
		for (int h = 0; h < ph; ++h) for (int w = 0; w < pw; ++w) {
			pad_in[i][h][w] = 0;
		}
	}
	for (int i = 0; i < ich; ++i) {
		for (int h = 0; h < height; ++h) for (int w = 0; w < width; ++w) {
			pad_in[i][h + p][w + p] = in[i][h][w];
		}
	}

	// conv
	for (int j = 0; j < och; ++j) {
		for (int h = 0; h < height; ++h) for (int w = 0; w < width; ++w) {
			out[j][h][w] = bias[j];
			for (int i = 0; i < ich; ++i) {
				for (int y = 0; y < KERNEL_SIZE; ++y) for (int x = 0; x < KERNEL_SIZE; ++x) {
					out[j][h][w] += pad_in[i][h + y][w + x] * weight[j][i][y][x];
				}
			}
		}
	}

	// relu
	for (int j = 0; j < och; ++j) {
		for (int h = 0; h < height; ++h) for (int w = 0; w < width; ++w) {
			out[j][h][w] = relu(out[j][h][w]);
		}
	}

}


void cnn_forward(
	double x_in[DIM],
	int num_ch[NUM_SEGMENTS][COUNT_CONV + 1],
	double weight[NUM_SEGMENTS][COUNT_CONV][MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE],
	double bias[NUM_SEGMENTS][COUNT_CONV][MAX_CH],
	double out[NUM_SEGMENTS][COUNT_CONV + 1][MAX_CH][HEIGHT][WIDTH],
	double flattened_out[MLP_INPUT_DIM]
) {

	int height = HEIGHT;
	int width = WIDTH;

	// 入力データを二次元マップへ変換
	for (int i = 0, ch = 0; ch < num_ch[0][0]; ++ch) {
		for (int h = 0; h < HEIGHT; ++h) for (int w = 0; w < WIDTH; ++w) {
			out[0][0][ch][h][w] = x_in[i++];
		}
	}

	for (int s = 0; s < NUM_SEGMENTS; ++s) {
		// Conv
		for (int c = 0; c < COUNT_CONV; ++c) {
			conv(out[s][c], out[s][c + 1], weight[s][c], bias[s][c], num_ch[s][c], num_ch[s][c + 1], height, width);
		}

		// Max Pooling
		int j = 0;
		for (int i = 0; i < num_ch[s][COUNT_CONV]; ++i) {
			for (int h = 0; h < height; h += POOL_SIZE) for (int w = 0; w < width; w += POOL_SIZE) {
				double max = -1e8;
				for (int y = 0; y < POOL_SIZE; ++y) for (int x = 0; x < POOL_SIZE; ++x) {
					if (out[s][COUNT_CONV][i][h + y][w + x] > max) max = out[s][COUNT_CONV][i][h + y][w + x];
				}

				// out[SEG - 1][CONV] -> flatten
				if (s == NUM_SEGMENTS - 1) { if (j < MLP_INPUT_DIM) flattened_out[j++] = max; }
				// cout[s][CONV] -> cout[s + 1][0]
				else out[s + 1][0][i][h / POOL_SIZE][w / POOL_SIZE] = max;
			}
		}

		height /= POOL_SIZE;
		width /= POOL_SIZE;
	}

}


void back_conv(
	double in[MAX_CH][HEIGHT][WIDTH],
	double out[MAX_CH][HEIGHT][WIDTH],
	double weight[MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE],
	double bias[MAX_CH],
	double indelta[MAX_CH][HEIGHT][WIDTH],
	double outdelta[MAX_CH][HEIGHT][WIDTH],
	double dweight[MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE],
	double dbias[MAX_CH],
	int ich,
	int och,
	int height,
	int width
) {

	int p = KERNEL_SIZE / 2;
	int ph = height + KERNEL_SIZE - 1;
	int pw = width + KERNEL_SIZE - 1;

	// 0 padding (input)
	for (int i = 0; i < ich; ++i) {
		for (int h = 0; h < ph; ++h) for (int w = 0; w < pw; ++w) {
			pad_in[i][h][w] = 0;
		}
	}
	for (int i = 0; i < ich; ++i) {
		for (int h = 0; h < height; ++h) for (int w = 0; w < width; ++w) {
			pad_in[i][h + p][w + p] = in[i][h][w];
		}
	}

	// 0 padding (delta)
	for (int i = 0; i < ich; ++i) {
		for (int h = 0; h < ph; ++h) for (int w = 0; w < pw; ++w) {
			pad_delta[i][h][w] = 0;
		}
	}

	// init dweight, dbias
	for (int j = 0; j < och; ++j) {
		dbias[j] = 0;
		for (int i = 0; i < ich; ++i) {
			for (int y = 0; y < KERNEL_SIZE; ++y) for (int x = 0; x < KERNEL_SIZE; ++x) {
				dweight[j][i][y][x] = 0;
			}
		}
	}

	// relu
	for (int i = 0; i < och; ++i) {
		for (int h = 0; h < height; ++h) for (int w = 0; w < width; ++w) {
			if (out[i][h][w] == 0) outdelta[i][h][w] = 0;
		}
	}

	// conv
	for (int j = 0; j < och; ++j) {
		for (int h = 0; h < height; ++h) for (int w = 0; w < width; ++w) {
			dbias[j] += outdelta[j][h][w];
			for (int i = 0; i < ich; ++i) {
				for (int y = 0; y < KERNEL_SIZE; ++y) for (int x = 0; x < KERNEL_SIZE; ++x) {
					pad_delta[i][h + y][w + x] += outdelta[j][h][w] * weight[j][i][y][x];
					dweight[j][i][y][x] += pad_in[i][h + y][w + x] * outdelta[j][h][w];
				}
			}
		}
	}

	// depadding delta
	for (int i = 0; i < ich; ++i) {
		for (int h = 0; h < height; ++h) for (int w = 0; w < width; ++w) {
			indelta[i][h][w] = pad_delta[i][h + p][w + p];
		}
	}

}


void cnn_backprop(
	double delta_fc[MLP_INPUT_DIM],
	int num_ch[NUM_SEGMENTS][COUNT_CONV + 1],
	double weight[NUM_SEGMENTS][COUNT_CONV][MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE],
	double bias[NUM_SEGMENTS][COUNT_CONV][MAX_CH],
	double out[NUM_SEGMENTS][COUNT_CONV + 1][MAX_CH][HEIGHT][WIDTH],
	double delta[NUM_SEGMENTS][COUNT_CONV + 1][MAX_CH][HEIGHT][WIDTH],
	double dweight[NUM_SEGMENTS][COUNT_CONV][MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE],
	double dbias[NUM_SEGMENTS][COUNT_CONV][MAX_CH]
) {

	int hsize = HEIGHT / pow(POOL_SIZE, NUM_SEGMENTS);
	int wsize = WIDTH / pow(POOL_SIZE, NUM_SEGMENTS);

	for (int s = NUM_SEGMENTS - 1; s >= 0; --s) {
		hsize *= POOL_SIZE;
		wsize *= POOL_SIZE;

		// Max Pooling
		int j = 0;
		for (int i = 0; i < num_ch[s][COUNT_CONV]; ++i) {
			for (int h = 0; h < hsize; h += POOL_SIZE) for (int w = 0; w < wsize; w += POOL_SIZE) {
				int maxh = h;
				int maxw = w;
				for (int y = 0; y < POOL_SIZE; ++y) for (int x = 0; x < POOL_SIZE; ++x) {
					delta[s][COUNT_CONV][i][h + y][w + x] = 0;
					if (out[s][COUNT_CONV][i][h + y][w + x] > out[s][COUNT_CONV][i][maxh][maxw]) {
						maxh = h + y;
						maxw = w + x;
					}
				}

				// mdelta -> cout[SEG - 1][CONV]
				if (s == NUM_SEGMENTS - 1) delta[s][COUNT_CONV][i][maxh][maxw] = delta_fc[j++];
				// cdelta[s + 1][0] -> cdelta[s][CONV]
				else delta[s][COUNT_CONV][i][maxh][maxw] = delta[s + 1][0][i][h / POOL_SIZE][w / POOL_SIZE];
			}
		}

		// Conv
		for (int c = COUNT_CONV - 1; c >= 0; --c) {
			back_conv(
				out[s][c], out[s][c + 1], weight[s][c], bias[s][c], 
				delta[s][c], delta[s][c + 1], dweight[s][c], dbias[s][c],
				num_ch[s][c], num_ch[s][c + 1], hsize, wsize
			);
		}
	}

	// 重み更新
	hsize = HEIGHT;
	wsize = WIDTH;
	for (int s = 0; s < NUM_SEGMENTS; ++s) {
		for (int c = 0; c < COUNT_CONV; ++c) {
			for (int j = 0; j < num_ch[s][c + 1]; ++j) {		// 出力ch
				int div = hsize * wsize;
				bias[s][c][j] -= LR * dbias[s][c][j] / div;
				for (int i = 0; i < num_ch[s][c]; ++i) {			// 入力ch
					for (int y = 0; y < KERNEL_SIZE; ++y) for (int x = 0; x < KERNEL_SIZE; ++x) {
						weight[s][c][j][i][y][x] -= LR * dweight[s][c][j][i][y][x] / div;
					}
				}
			}
		}
		hsize /= POOL_SIZE;
		wsize /= POOL_SIZE;
	}

}


