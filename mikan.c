#include "mikan.h"
#include <math.h>
#include <stdlib.h>


static const double EPS = 0.000001;


void mikan_init(
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double emlp_weight[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double emlp_bias[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double beta[KAN_NUM_LAYERS],
	double gamma[KAN_NUM_LAYERS]
) {

	if (!NO_WEIGHT_AND_BASIS) {

		// wbはXavier初期化
		for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
			double xavier_weight = sqrt(6.0 / (num_nodes[l] + num_nodes[l + 1]));
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				for (int i = 0; i < num_nodes[l]; ++i) {
					wb[l][j][i] = xavier_weight * (((double)rand() / RAND_MAX) * 2 - 1);
					//wb[l][j][i] = 0;
				}
			}
		}

		// wsは1で初期化
		for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				for (int i = 0; i < num_nodes[l]; ++i) {
					ws[l][j][i] = 1;
				}
			}
		}

	}

	// EdgeMLP
	for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			for (int i = 0; i < num_nodes[l]; ++i) {
				if (KANTYPE == 0) emlp_init(emlp_weight[l][j][i], emlp_bias[l][j][i]);
				else if (KANTYPE == 1) emlpk_init(emlp_weight[l][j][i], emlp_bias[l][j][i]);
				else emlps_init(emlp_weight[l][j][i], emlp_bias[l][j][i]);
			}
		}
	}

	// beta, gamma
	for (int l = 0; l < KAN_NUM_LAYERS; ++l) {
		beta[l] = 0;
		gamma[l] = 1;
	}

}

void mikan_forward(
	double x[DIM],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double emlp_weight[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double emlp_bias[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double emlp_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES],
	double bnet[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double mean[KAN_NUM_LAYERS],
	double var[KAN_NUM_LAYERS],
	double beta[KAN_NUM_LAYERS],
	double gamma[KAN_NUM_LAYERS]
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
				// EdgeMLP
				if (KANTYPE == 0) emlp_forward(out[l][i], emlp_weight[l][j][i], emlp_bias[l][j][i], emlp_out[l][j][i]);
				else if (KANTYPE == 1) emlpk_forward(out[l][i], emlp_weight[l][j][i], emlp_bias[l][j][i], emlp_out[l][j][i]);
				else emlps_forward(out[l][i], emlp_weight[l][j][i], emlp_bias[l][j][i], emlp_out[l][j][i]);

				double res = emlp_out[l][j][i][EMLP_NUM_LAYERS - 1][0];

				if (NO_WEIGHT_AND_BASIS) out[l + 1][j] += res;
				else out[l + 1][j] += wb[l][j][i] * silu_out[l][i] + ws[l][j][i] * res;
			}
		}

		// Layer Norm
		if (LAYER_NORM) mikan_layer_norm_forward(num_nodes[l + 1], out[l + 1], bnet[l + 1], &mean[l + 1], &var[l + 1], beta[l + 1], gamma[l + 1]);
		//if (l < KAN_NUM_LAYERS - 2 && LAYER_NORM) mikan_layer_norm_forward(num_nodes[l + 1], out[l + 1], bnet[l + 1], & mean[l + 1], &var[l + 1], beta[l + 1], gamma[l + 1]);
	}

	// 最終層でsoftmax
	const int last_layer = KAN_NUM_LAYERS - 1;
	softmax(out[last_layer], num_nodes[last_layer]);

}

void mikan_backprop(
	bool t[NUM_CLASSES],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double emlp_weight[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double emlp_bias[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double emlp_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double emlp_delta[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double delta[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES],
	double bnet[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double mean[KAN_NUM_LAYERS],
	double var[KAN_NUM_LAYERS],
	double beta[KAN_NUM_LAYERS],
	double gamma[KAN_NUM_LAYERS]
) {

	const int last_layer = KAN_NUM_LAYERS - 1;

	// output layer (squared err or cross-entropy err)
	for (int i = 0; i < NUM_CLASSES; ++i) {
		delta[last_layer][i] = out[last_layer][i] - (double)t[i];
	}

	// hidden layer
	for (int l = KAN_NUM_LAYERS - 2; l > 0; --l) {
		// Layer Norm
		if (LAYER_NORM) mikan_layer_norm_backprop(num_nodes[l + 1], out[l + 1], bnet[l + 1], delta[l + 1], mean[l + 1], var[l + 1], &beta[l + 1], &gamma[l + 1]);

		for (int i = 0; i < num_nodes[l]; ++i) {
			delta[l][i] = 0;
			const double sig_out = sigmoid(out[l][i]);
			const double dsilu = sig_out + out[l][i] * sig_out * (1 - sig_out);
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				// EdgeMLP
				if (KANTYPE == 0) emlp_backprop(delta[l + 1][j], emlp_weight[l][j][i], emlp_bias[l][j][i], emlp_out[l][j][i], emlp_delta[l][j][i]);
				else if (KANTYPE == 1) emlpk_backprop(delta[l + 1][j], emlp_weight[l][j][i], emlp_bias[l][j][i], emlp_out[l][j][i], emlp_delta[l][j][i]);
				else emlps_backprop(delta[l + 1][j], emlp_weight[l][j][i], emlp_bias[l][j][i], emlp_out[l][j][i], emlp_delta[l][j][i]);

				double d = emlp_delta[l][j][i][0][0];

				if (NO_WEIGHT_AND_BASIS) delta[l][i] += d * delta[l + 1][j];
				delta[l][i] += (wb[l][j][i] * dsilu + ws[l][j][i] * d) * delta[l + 1][j];
			}
		}
	}

	if (!NO_WEIGHT_AND_BASIS) {
		// update wb, ws
		for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				for (int i = 0; i < num_nodes[l]; ++i) {
					wb[l][j][i] -= KAN_LR * (delta[l + 1][j] * silu_out[l][i]);
					//ws[l][j][i] -= KAN_LR * (delta[l + 1][j] * emlp_out[l][j][i][EMLP_NUM_LAYERS - 1][0]);
				}
			}
		}
	}

}

void mikan_layer_norm_forward(
	int num_nodes,
	double out[KAN_MAX_NODES],
	double bnet[KAN_MAX_NODES],
	double* mean,
	double* var,
	double beta,
	double gamma
) {

	// save vals before norm
	for (int i = 0; i < num_nodes; ++i) bnet[i] = out[i];

	// mean
	double sum = 0;
	for (int i = 0; i < num_nodes; ++i) {
		sum += out[i];
	}
	*mean = sum / num_nodes;

	// var
	sum = 0;
	for (int i = 0; i < num_nodes; ++i) {
		sum += (out[i] - *mean) * (out[i] - *mean);
	}
	*var = sum / num_nodes;

	// normalize
	for (int i = 0; i < num_nodes; ++i) {
		out[i] = (out[i] - *mean) / sqrt(*var + EPS);
	}

	// scale and shift
	for (int i = 0; i < num_nodes; ++i) {
		out[i] = gamma * out[i] + beta;
	}

}


void mikan_layer_norm_backprop(
	int num_nodes,
	double out[KAN_MAX_NODES],
	double bnet[KAN_MAX_NODES],
	double delta[KAN_MAX_NODES],
	double mean,
	double var,
	double* beta,
	double* gamma
) {

	double inv_std = 1 / sqrt(var + EPS);

	double dxh[KAN_MAX_NODES];
	for (int i = 0; i < num_nodes; ++i) dxh[i] = delta[i] * (*gamma);

	// var
	double dvar = 0;
	for (int i = 0; i < num_nodes; ++i) {
		dvar += dxh[i] * (bnet[i] - mean) * (-0.5) * pow(var + EPS, -1.5);
	}

	// mean
	double dmean = 0;
	for (int i = 0; i < num_nodes; ++i) {
		dmean += -2 * (bnet[i] - mean);
	}
	dmean *= dvar / num_nodes;
	for (int i = 0; i < num_nodes; ++i) {
		dmean += dxh[i] * (-inv_std);
	}

	// beta, gamma
	double dbeta = 0.0;
	double dgamma = 0.0;
	for (int i = 0; i < num_nodes; ++i) {
		dbeta += delta[i];
		dgamma += delta[i] * (bnet[i] - mean) * inv_std;
	}

	// delta
	for (int i = 0; i < num_nodes; ++i) {
		delta[i] = dxh[i] * inv_std + dvar * 2 * (bnet[i] - mean) / num_nodes + dmean / num_nodes;
	}

	// update beta, gamma
	*beta += -KAN_LR * dbeta / num_nodes;
	*gamma += -KAN_LR * dgamma / num_nodes;

}
