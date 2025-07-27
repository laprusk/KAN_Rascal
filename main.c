#include "mlp.h"
#include "cnn.h"
#include "kan.h"
#include "dataset.h"
#include "util.h"
#include <stdio.h>
#include <stdbool.h>
#include <time.h>


// 定数
const int EPOCH_MAX = 10;
const bool CNN = 0;


// MLP
int mlp_num_nodes[MLP_NUM_LAYERS] = {MLP_INPUT_DIM, DIM, NUM_CLASSES};
const Activation HIDDEN_ACTIVATION = RELU;
const Activation OUTPUT_ACTIVATION = SOFTMAX;
double mlp_weight[MLP_NUM_LAYERS - 1][MLP_MAX_NODES][MLP_MAX_NODES];
double mlp_bias[MLP_NUM_LAYERS - 1][MLP_MAX_NODES];
// online
double mlp_out[MLP_NUM_LAYERS][MLP_MAX_NODES];
double mlp_delta[MLP_NUM_LAYERS][MLP_MAX_NODES];
// mini-batch


//CNN
int num_ch[NUM_SEGMENTS][COUNT_CONV + 1] = {
	{CHANNEL, MAX_CH / 4, MAX_CH / 2},
	{MAX_CH / 2, MAX_CH, MAX_CH}
};
// online
double cnn_weight[NUM_SEGMENTS][COUNT_CONV][MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE];
double cnn_bias[NUM_SEGMENTS][COUNT_CONV][MAX_CH];
double cnn_out[NUM_SEGMENTS][COUNT_CONV + 1][MAX_CH][HEIGHT][WIDTH];
double cnn_delta[NUM_SEGMENTS][COUNT_CONV + 1][MAX_CH][HEIGHT][WIDTH];
double cnn_dweight[NUM_SEGMENTS][COUNT_CONV][MAX_CH][MAX_CH][KERNEL_SIZE][KERNEL_SIZE];
double cnn_dbias[NUM_SEGMENTS][COUNT_CONV][MAX_CH];
// mini-batch


// KAN
int kan_num_nodes[KAN_NUM_LAYERS] = { KAN_INPUT_DIM, 16, NUM_CLASSES };
const KANFunction func_type = GRBF;
double knots[NUM_KNOTS];
double coeff[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][NUM_CP];
double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES];
double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES];
// online
double kan_out[KAN_NUM_LAYERS][KAN_MAX_NODES];
double kan_delta[KAN_NUM_LAYERS][KAN_MAX_NODES];
double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES];
double spline_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES];
double basis_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP];


// Dataset
double train_data[NUM_TRAINS][DIM];
double test_data[NUM_TESTS][DIM];
int train_label[NUM_TRAINS];
int test_label[NUM_TESTS];


void train_mlp() {

	double x[DIM];
	bool tk[NUM_CLASSES];
	int train_order[NUM_TRAINS];

	if (CNN) printf("train CNN...\n\n");
	else printf("train MLP...\n\n");

	// init weight
	mlp_init(mlp_num_nodes, mlp_weight, mlp_bias, HIDDEN_ACTIVATION);
	if (CNN) cnn_init(num_ch, cnn_weight, cnn_bias);

	// train loop
	for (int ep = 0; ep < EPOCH_MAX; ++ep) {
		// shuffle dataset order
		for (int i = 0; i < NUM_TRAINS; ++i) train_order[i] = i;
		shuffle(train_order, NUM_TRAINS);

		// train
		for (int t = 0; t < NUM_TRAINS; ++t) {
			// make input & label
			int i = train_order[t];
			memcpy(x, train_data[i], sizeof(train_data[i]));
			convert_one_hot(train_label[i], tk);

			// forward & backprop
			if (CNN) cnn_forward(x, num_ch, cnn_weight, cnn_bias, cnn_out, x);
			mlp_forward(
				x, mlp_num_nodes, mlp_weight, mlp_bias, mlp_out,
				HIDDEN_ACTIVATION, OUTPUT_ACTIVATION
			);
			mlp_backprop(
				tk, mlp_num_nodes, mlp_weight, mlp_bias, mlp_out, mlp_delta,
				HIDDEN_ACTIVATION, OUTPUT_ACTIVATION, CNN
			);
			if (CNN) cnn_backprop(
				mlp_delta[0], num_ch, cnn_weight, cnn_bias, cnn_out, cnn_delta, cnn_dweight, cnn_dbias
			);
		}

		// evaluate test
		int count = 0;
		for (int t = 0; t < NUM_TESTS; ++t) {
			int i = t;
			memcpy(x, test_data[i], sizeof(test_data[i]));

			// forward only
			if (CNN) cnn_forward(x, num_ch, cnn_weight, cnn_bias, cnn_out, x);
			mlp_forward(
				x, mlp_num_nodes, mlp_weight, mlp_bias, mlp_out,
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
		memcpy(x, train_data[i], sizeof(train_data[i]));

		// forward only
		if (CNN) cnn_forward(x, num_ch, cnn_weight, cnn_bias, cnn_out, x);
		mlp_forward(
			x, mlp_num_nodes, mlp_weight, mlp_bias, mlp_out,
			HIDDEN_ACTIVATION, OUTPUT_ACTIVATION
		);
		if (mlp_is_collect(mlp_out[MLP_NUM_LAYERS - 1], train_label[i])) ++count;
	}
	printf("Train: %.3f\n\n", (double)count / NUM_TRAINS);

}

void train_kan() {

	double x[DIM];
	bool tk[NUM_CLASSES];
	int train_order[NUM_TRAINS];

	printf("train KAN...\n\n");

	// init weight
	kan_init(kan_num_nodes, wb, ws, coeff, knots, func_type);

	// timer Start
	const clock_t start_clock = clock();

	// train loop
	for (int ep = 0; ep < EPOCH_MAX; ++ep) {
		// shuffle dataset order
		for (int i = 0; i < NUM_TRAINS; ++i) train_order[i] = i;
		shuffle(train_order, NUM_TRAINS);

		// train
		for (int t = 0; t < NUM_TRAINS; ++t) {
			// make input & label
			int i = train_order[t];
			memcpy(x, train_data[i], sizeof(train_data[i]));
			convert_one_hot(train_label[i], tk);

			// forward & backprop
			kan_forward(x, kan_num_nodes, wb, ws, coeff, knots, kan_out, silu_out, spline_out, basis_out, func_type);
			kan_backprop(tk, kan_num_nodes, wb, ws, coeff, knots, kan_out, kan_delta, silu_out, spline_out, basis_out, func_type);
		}

		// evaluate test
		int count = 0;
		for (int t = 0; t < NUM_TESTS; ++t) {
			int i = t;
			memcpy(x, test_data[i], sizeof(test_data[i]));

			// forward only
			kan_forward(x, kan_num_nodes, wb, ws, coeff, knots, kan_out, silu_out, spline_out, basis_out, func_type);
			if (mlp_is_collect(kan_out[KAN_NUM_LAYERS - 1], test_label[i])) ++count;
		}
		const double sec = (double)(clock() - start_clock) / CLOCKS_PER_SEC;
		printf("Epoch %d: %.3f (%.3fs)\n", ep, (double)count / NUM_TESTS, sec);
	}

	// evaluate train
	int count = 0;
	for (int t = 0; t < NUM_TRAINS; ++t) {
		int i = t;
		memcpy(x, train_data[i], sizeof(train_data[i]));

		// forward only
		kan_forward(x, kan_num_nodes, wb, ws, coeff, knots, kan_out, silu_out, spline_out, basis_out, func_type);
		if (mlp_is_collect(kan_out[KAN_NUM_LAYERS - 1], train_label[i])) ++count;
	}
	printf("Train: %.3f\n\n", (double)count / NUM_TRAINS);

}


int main() {

	// 乱数初期化
	//srand((unsigned int)time(NULL));
	//srand(3);
	
	// データセット読み込み
	load_dataset(train_data, test_data, train_label, test_label);

	//train_mlp();
	train_kan();

	return 0;
}
