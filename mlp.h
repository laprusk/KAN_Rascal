#pragma once

#include "dataset.h"
#include "activation.h"
#include <stdbool.h>

#define MLP_NUM_LAYERS 3
#define MLP_MAX_NODES DIM
#define LR 0.01

void mlp_initialize(
	int num_nodes[],
	double weight[][MLP_MAX_NODES][MLP_MAX_NODES],
	double bias[][MLP_MAX_NODES],
	Activation hidden_activation
);
void mlp_forward(
	double x[],
	int num_nodes[],
	double out[][MLP_MAX_NODES],
	double weight[][MLP_MAX_NODES][MLP_MAX_NODES],
	double bias[][MLP_MAX_NODES],
	Activation hidden_activation,
	Activation output_activation
);
void mlp_backprop(
	bool t[],
	int num_nodes[],
	double out[][MLP_MAX_NODES],
	double delta[][MLP_MAX_NODES],
	double weight[][MLP_MAX_NODES][MLP_MAX_NODES],
	double bias[][MLP_MAX_NODES],
	Activation hidden_activation,
	Activation output_activation
);
bool mlp_is_collect(double output[], int label);

