#pragma once

#include "kan.h"
#include "edge_mlp.h"

void mikan_init(
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	int edge_mlp_num_nodes[EDGE_MLP_NUM_LAYERS],
	double edge_mlp_weight[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EDGE_MLP_NUM_LAYERS - 1][EDGE_MLP_MAX_NODES][EDGE_MLP_MAX_NODES],
	double edge_mlp_bias[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EDGE_MLP_NUM_LAYERS - 1][EDGE_MLP_MAX_NODES]
);
void mikan_forward(
	double x[DIM],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	int edge_mlp_num_nodes[EDGE_MLP_NUM_LAYERS],
	double edge_mlp_weight[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EDGE_MLP_NUM_LAYERS - 1][EDGE_MLP_MAX_NODES][EDGE_MLP_MAX_NODES],
	double edge_mlp_bias[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EDGE_MLP_NUM_LAYERS - 1][EDGE_MLP_MAX_NODES],
	double edge_mlp_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EDGE_MLP_NUM_LAYERS][EDGE_MLP_MAX_NODES],
	Activation edge_mlp_activation,
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES]
);
void mikan_backprop(
	bool t[NUM_CLASSES],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	int edge_mlp_num_nodes[EDGE_MLP_NUM_LAYERS],
	double edge_mlp_weight[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EDGE_MLP_NUM_LAYERS - 1][EDGE_MLP_MAX_NODES][EDGE_MLP_MAX_NODES],
	double edge_mlp_bias[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EDGE_MLP_NUM_LAYERS - 1][EDGE_MLP_MAX_NODES],
	double edge_mlp_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EDGE_MLP_NUM_LAYERS][EDGE_MLP_MAX_NODES],
	double edge_mlp_delta[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EDGE_MLP_NUM_LAYERS][EDGE_MLP_MAX_NODES],
	Activation edge_mlp_activation,
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double delta[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES]
);
