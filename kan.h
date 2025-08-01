#pragma once

#include "dataset.h"

#define KAN_NUM_LAYERS 3
#define KAN_INPUT_DIM DIM
#define KAN_MAX_NODES DIM
#define GRID_SIZE 5
#define SPLINE_ORDER 3
#define NUM_CP (GRID_SIZE + SPLINE_ORDER)
#define NUM_KNOTS (GRID_SIZE + 1 + SPLINE_ORDER * 2)
#define GRID_MIN -1.0
#define GRID_MAX 1.0
#define KAN_LR 0.005

// 動作定義
#define NO_WEIGHT_AND_BASIS 1			// SiLU基底関数と重みを使用しない

typedef enum {
	B_SPLINE,			// Original
	GRBF,					// Fast-KAN
	RSWAF,				// Faster-KAN
	RELU_KAN			// ReLU-KAN
} KANFunction;

void kan_init(
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double coeff[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][NUM_CP],
	double knots[NUM_KNOTS],
	double phase_low[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	double phase_height[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	KANFunction func_type
);
void kan_forward(
	double x[DIM],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double coeff[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][NUM_CP],
	double knots[NUM_KNOTS],
	double phase_low[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	double phase_height[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES],
	double spline_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double basis_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	KANFunction func_type
);
void kan_backprop(
	bool t[NUM_CLASSES],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double coeff[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][NUM_CP],
	double knots[NUM_KNOTS],
	double phase_low[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	double phase_height[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double delta[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES],
	double spline_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double basis_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	KANFunction func_type
);
