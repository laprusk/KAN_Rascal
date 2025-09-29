#include "mikan2.h"
#include "edge_mlp.h"
#include <stdlib.h>


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
) {

	// 入力を0層の出力へコピー
	for (int i = 0; i < DIM; ++i) {
		out[0][i] = x[i];
	}

	for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
		for (int i = 0; i < num_nodes[l]; ++i) {
			silu_out[l][i] = silu(out[l][i]);
		}
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			out[l + 1][j] = 0;
			for (int i = 0; i < num_nodes[l]; ++i) {
				spline_out[l][j][i] = emlp_forward(out[l][i], emlp_num_nodes, emlp_weight[l][j][i], emlp_bias[l][j][i], emlp_out[l][j][i]);

				if (NO_WEIGHT_AND_BASIS) out[l + 1][j] += spline_out[l][j][i];
				else out[l + 1][j] += wb[l][j][i] * silu_out[l][i] + ws[l][j][i] * spline_out[l][j][i];
			}
		}

		// Layer Norm
		if (LAYER_NORM) kan_layer_norm_forward(num_nodes[l + 1], out[l + 1], &mean[l + 1], &var[l + 1]);
	}

	// 最終層でsoftmax
	const int last_layer = KAN_NUM_LAYERS - 1;
	softmax(out[last_layer], num_nodes[last_layer]);

}

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
) {

	const int last_layer = KAN_NUM_LAYERS - 1;

	// output layer (squared err or cross-entropy err)
	for (int i = 0; i < NUM_CLASSES; ++i) {
		delta[last_layer][i] = out[last_layer][i] - (double)t[i];
	}

	// hidden layer
	for (int l = KAN_NUM_LAYERS - 2; l > 0; --l) {
		// Layer Norm
		if (LAYER_NORM) kan_layer_norm_backprop(num_nodes[l + 1], out[l + 1], delta[l + 1], mean[l + 1], var[l + 1]);

		for (int i = 0; i < num_nodes[l]; ++i) {
			delta[l][i] = 0;
			const double sig_out = sigmoid(out[l][i]);
			const double dsilu = sig_out + out[l][i] * sig_out * (1 - sig_out);
			//const double dsilu = 1;
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				const double dspline = emlp_backprop(delta[l + 1][j], emlp_num_nodes, emlp_weight[l][j][i], emlp_bias[l][j][i], emlp_out[l][j][i], emlp_delta[l][j][i]);
				//const double dspline = 1;

				if (NO_WEIGHT_AND_BASIS) delta[l][i] += dspline * delta[l + 1][j];
				delta[l][i] += (wb[l][j][i] * dsilu + ws[l][j][i] * dspline) * delta[l + 1][j];
			}
		}
	}

	if (!NO_WEIGHT_AND_BASIS) {
		// update wb, ws
		for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				for (int i = 0; i < num_nodes[l]; ++i) {
					wb[l][j][i] -= KAN_LR * (delta[l + 1][j] * silu_out[l][i]);
					ws[l][j][i] -= KAN_LR * (delta[l + 1][j] * spline_out[l][j][i]);
				}
			}
		}
	}

}
