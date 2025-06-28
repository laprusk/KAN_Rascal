#include "mlp.h"
#include "dataset.h"
#include "util.h"
#include <stdio.h>
#include <stdbool.h>


// 定数
const int EPOCH_MAX = 10;


// MLP
int mlp_num_nodes[MLP_NUM_LAYERS] = {DIM, DIM, NUM_CLASSES};
const Activation HIDDEN_ACTIVATION = RELU;
const Activation OUTPUT_ACTIVATION = SOFTMAX;
// online
double mlp_out[MLP_NUM_LAYERS][MLP_MAX_NODES];
double mlp_delta[MLP_NUM_LAYERS][MLP_MAX_NODES];
double mlp_weight[MLP_NUM_LAYERS - 1][MLP_MAX_NODES][MLP_MAX_NODES];
double mlp_bias[MLP_NUM_LAYERS - 1][MLP_MAX_NODES];
// mini-batch


//CNN

// online

// mini-batch


// Dataset
double train_data[NUM_TRAINS][DIM];
double test_data[NUM_TESTS][DIM];
int train_label[NUM_TRAINS];
int test_label[NUM_TESTS];


void train_mlp() {

	bool tk[NUM_CLASSES];
	int train_order[NUM_TRAINS];

	mlp_initialize(mlp_num_nodes, mlp_weight, mlp_bias, HIDDEN_ACTIVATION);

	printf("train MLP...\n\n");
	for (int ep = 0; ep < EPOCH_MAX; ++ep) {
		// shuffle dataset order
		for (int i = 0; i < NUM_TRAINS; ++i) train_order[i] = i;
		shuffle(train_order, NUM_TRAINS);

		// train
		for (int t = 0; t < NUM_TRAINS; ++t) {
			int i = train_order[t];
			convert_one_hot(train_label[i], tk);

			mlp_forward(
				train_data[i], mlp_num_nodes, mlp_out, mlp_weight, mlp_bias,
				HIDDEN_ACTIVATION, OUTPUT_ACTIVATION
			);
			mlp_backprop(
				tk, mlp_num_nodes, mlp_out, mlp_delta, mlp_weight, mlp_bias,
				HIDDEN_ACTIVATION, OUTPUT_ACTIVATION
			);
		}

		// evaluate test
		int count = 0;
		for (int t = 0; t < NUM_TESTS; ++t) {
			int i = t;
			mlp_forward(
				test_data[i], mlp_num_nodes, mlp_out, mlp_weight, mlp_bias,
				HIDDEN_ACTIVATION, OUTPUT_ACTIVATION
			);
			if (mlp_is_collect(mlp_out[MLP_NUM_LAYERS - 1], test_label[i])) ++count;
		}
		printf("Epoch %d: %.3f\n", ep, (double)count / NUM_TESTS);
	}

	// evaluate train
	int count = 0;
	for (int t = 0; t < NUM_TRAINS; ++t) {
		int i = t;
		mlp_forward(
			train_data[i], mlp_num_nodes, mlp_out, mlp_weight, mlp_bias,
			HIDDEN_ACTIVATION, OUTPUT_ACTIVATION
		);
		if (mlp_is_collect(mlp_out[MLP_NUM_LAYERS - 1], train_label[i])) ++count;
	}
	printf("Train: %.3f\n\n", (double)count / NUM_TRAINS);

}


int main() {
	
	// データセット読み込み
	load_dataset(train_data, test_data, train_label, test_label);

	train_mlp();

	return 0;
}
