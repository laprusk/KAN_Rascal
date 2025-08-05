#pragma once

#include "activation.h"

#define EDGE_MLP_NUM_LAYERS 3
#define EDGE_MLP_MAX_NODES 4

double edge_mlp_forward(
	double x,
	int num_nodes[EDGE_MLP_NUM_LAYERS],
	double weight[EDGE_MLP_NUM_LAYERS - 1][EDGE_MLP_MAX_NODES][EDGE_MLP_MAX_NODES],
	double bias[EDGE_MLP_NUM_LAYERS - 1][EDGE_MLP_MAX_NODES],
	double out[EDGE_MLP_NUM_LAYERS][EDGE_MLP_MAX_NODES],
	Activation activation
);
void edge_mlp_backprop(
	double kan_delta,
	int num_nodes[EDGE_MLP_NUM_LAYERS],
	double weight[EDGE_MLP_NUM_LAYERS - 1][EDGE_MLP_MAX_NODES][EDGE_MLP_MAX_NODES],
	double bias[EDGE_MLP_NUM_LAYERS - 1][EDGE_MLP_MAX_NODES],
	double out[EDGE_MLP_NUM_LAYERS][EDGE_MLP_MAX_NODES],
	double delta[EDGE_MLP_NUM_LAYERS][EDGE_MLP_MAX_NODES],
	Activation activation
);
