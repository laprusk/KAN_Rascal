#pragma once

#include "kan.h"
#include "edge_mlp.h"

double mikan_forward2(
	double x[DIM],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	int emlp_num_nodes[EMLP_NUM_LAYERS],
	double emlp_weight[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double emlp_bias[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double emlp_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES],
	double spline_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double mean[KAN_NUM_LAYERS],
	double var[KAN_NUM_LAYERS]
);
double mikan_backprop2(
	bool t[NUM_CLASSES],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	int emlp_num_nodes[EMLP_NUM_LAYERS],
	double emlp_weight[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double emlp_bias[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double emlp_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double emlp_delta[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double delta[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES],
	double spline_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double mean[KAN_NUM_LAYERS],
	double var[KAN_NUM_LAYERS]
);
