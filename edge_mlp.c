#include "edge_mlp.h"
#include "kan.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>


// 活性化関数適用後の出力
double aout[EMLP_NUM_LAYERS][EMLP_MAX_NODES];


void emlp_init(
	int num_nodes[EMLP_NUM_LAYERS],
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES]
) {
	int l;

	// hidden layer
	for (l = 0; l < EMLP_NUM_LAYERS - 2; ++l) {
		// He初期化
		double he_wp = 2 * sqrt(6.0 / num_nodes[l]);
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			bias[l][j] = 0;
			for (int i = 0; i < num_nodes[l]; ++i) {
				weight[l][j][i] = ((double)rand() / RAND_MAX - 0.5) * he_wp;
				//woi[l][j][i] = randn() / sqrt(NodeN[l] / 2);
			}
		}
	}

	// output layer
	for (int j = 0; j < num_nodes[l + 1]; ++j) {
		bias[l][j] = 0;
		for (int i = 0; i < num_nodes[l]; ++i) {
			weight[l][j][i] = (double)rand() / (RAND_MAX)-0.5;
		}
	}

}


void emlp_forward(
	double x,
	int num_nodes[EMLP_NUM_LAYERS],
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
) {
	int l;

	// 入力を0層の出力へコピー
	out[0][0] = x;
	aout[0][0] = x;

	// forward
	for (l = 1; l < EMLP_NUM_LAYERS; ++l) {
		for (int j = 0; j < num_nodes[l]; ++j) {
			out[l][j] = bias[l - 1][j];
			for (int i = 0; i < num_nodes[l - 1]; ++i) {
				out[l][j] += weight[l - 1][j][i] * aout[l - 1][i];
			}
			
			// activation
			if (l != EMLP_NUM_LAYERS - 1) aout[l][j] = silu(out[l][j]);
		}
	}

}


void emlp_backprop(
	double kan_delta,
	int num_nodes[EMLP_NUM_LAYERS],
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double delta[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
) {

	int l;

	// 最終層のデルタ
	delta[EMLP_NUM_LAYERS - 1][0] = kan_delta;

	// hidden layer
	for (l = EMLP_NUM_LAYERS - 2; l > 0; --l) {
		for (int i = 0; i < num_nodes[l]; ++i) {
			delta[l][i] = 0;
			if (out[l][i] != 0) {			// 出力が0なら、活性化関数によらず誤差信号も0
				for (int j = 0; j < num_nodes[l + 1]; ++j) {
					delta[l][i] += delta[l + 1][j] * weight[l][j][i];
				}

				if (l != 0) {
					double sig = sigmoid(out[l][i]);
					delta[l][i] *= sig + out[l][i] * (1 - sig) * sig;
				}
			}
		}
	}

	// 重み・バイアスの更新
	for (l = 0; l < EMLP_NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			bias[l][j] -= KAN_LR * delta[l + 1][j];
			for (int i = 0; i < num_nodes[l]; ++i) {
				weight[l][j][i] -= KAN_LR * delta[l + 1][j] * out[l][i];
			}
		}
	}

}
