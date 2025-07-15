#pragma once

#include "dataset.h"
#include "activation.h"
#include <stdbool.h>

#define MLP_NUM_LAYERS 3
#define MLP_INPUT_DIM DIM
#define MLP_MAX_NODES DIM
#define MLP_LR 0.01

void mlp_init(
	int num_nodes[MLP_NUM_LAYERS],
	double weight[MLP_NUM_LAYERS - 1][MLP_MAX_NODES][MLP_MAX_NODES],
	double bias[MLP_NUM_LAYERS - 1][MLP_MAX_NODES],
	Activation hidden_activation
);
void mlp_forward(
	double x[DIM],
	int num_nodes[MLP_NUM_LAYERS],
	double weight[MLP_NUM_LAYERS - 1][MLP_MAX_NODES][MLP_MAX_NODES],
	double bias[MLP_NUM_LAYERS - 1][MLP_MAX_NODES],
	double out[MLP_NUM_LAYERS][MLP_MAX_NODES],
	Activation hidden_activation,
	Activation output_activation
);
void mlp_backprop(
	bool t[NUM_CLASSES],
	int num_nodes[MLP_NUM_LAYERS],
	double weight[MLP_NUM_LAYERS - 1][MLP_MAX_NODES][MLP_MAX_NODES],
	double bias[MLP_NUM_LAYERS - 1][MLP_MAX_NODES],
	double out[MLP_NUM_LAYERS][MLP_MAX_NODES],
	double delta[MLP_NUM_LAYERS][MLP_MAX_NODES],
	Activation hidden_activation,
	Activation output_activation,
	bool is_delta_0layer
);
bool mlp_is_collect(double output[], int label);
