#pragma once

#include "activation.h"

#define EMLP_NUM_LAYERS 3
#define EMLP_MAX_NODES 4

void emlp_init(
	int num_nodes[EMLP_NUM_LAYERS],
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES]
);
void emlp_forward(
	double x,
	int num_nodes[EMLP_NUM_LAYERS],
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
);
void emlp_backprop(
	double kan_delta,
	int num_nodes[EMLP_NUM_LAYERS],
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double delta[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
);
