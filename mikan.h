#pragma once

#include "kan.h"
#include "edge_mlp.h"

// 0: EdgeMLP, 1: EdgeMLP-k, 2: EdgeMLP-spline
#define KANTYPE 0

void mikan_init(
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double emlp_weight[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double emlp_bias[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double beta[KAN_NUM_LAYERS],
	double gamma[KAN_NUM_LAYERS]
);
void mikan_forward(
	double x[DIM],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double emlp_weight[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double emlp_bias[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double emlp_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double emlp_out_ba[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES],
	double bnet[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double mean[KAN_NUM_LAYERS],
	double var[KAN_NUM_LAYERS],
	double beta[KAN_NUM_LAYERS],
	double gamma[KAN_NUM_LAYERS]
);
void mikan_backprop(
	bool t[NUM_CLASSES],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double emlp_weight[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double emlp_bias[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double emlp_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double emlp_out_ba[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double emlp_delta[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double delta[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES],
	double bnet[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double mean[KAN_NUM_LAYERS],
	double var[KAN_NUM_LAYERS],
	double beta[KAN_NUM_LAYERS],
	double gamma[KAN_NUM_LAYERS]
);
void mikan_layer_norm_forward(
	int num_nodes,
	double out[KAN_MAX_NODES],
	double bnet[KAN_MAX_NODES],
	double* mean,
	double* var,
	double beta,
	double gamma
);
void mikan_layer_norm_backprop(
	int num_nodes,
	double out[KAN_MAX_NODES],
	double bnet[KAN_MAX_NODES],
	double delta[KAN_MAX_NODES],
	double mean,
	double var,
	double* beta,
	double* gamma
);
