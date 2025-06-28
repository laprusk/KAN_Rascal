#pragma once

#include "dataset.h"
#include "activation.h"
#include <stdbool.h>

#define MLP_MAX_LAYERS 2
#define MLP_MAX_NODES DIM

typedef struct {
	int num_layers;
	int num_nodes[MLP_MAX_LAYERS + 1];
	double learning_rate;
	Activation hidden_activation;
	Activation out_activation;
	double weight[MLP_MAX_LAYERS][MLP_MAX_NODES][MLP_MAX_NODES];
	double bias[MLP_MAX_LAYERS][MLP_MAX_NODES];
	double out[MLP_MAX_LAYERS + 1][MLP_MAX_NODES];
	double delta[MLP_MAX_LAYERS + 1][MLP_MAX_NODES];
} MLP;

void mlp_initialize(MLP* model);
void mlp_forward(MLP* model, double* input, int dim_input);
void mlp_backprop(MLP* model, bool* tk, int num_classes);
bool mlp_is_collect(MLP* model, int label);
