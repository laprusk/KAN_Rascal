#pragma once

#include "activation.h"

#define EMLP_D 4
#define EMLP_K 1
#define EMLP_NUM_LAYERS 3
#define EMLP_MAX_NODES EMLP_D
#define EMLPS_BOUND 0.6

void emlp_init(
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES]
);
double emlp_forward(
	double x,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
);
double emlp_backprop(
	double kan_delta,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double delta[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
);
void emlpk_init(
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES]
);
double emlpk_forward(
	double x,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
);
double emlpk_backprop(
	double kan_delta,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double delta[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
);
void emlps_init(
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES]
);
double emlps_forward(
	double x,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
);
double emlps_backprop(
	double kan_delta,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double delta[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
);
