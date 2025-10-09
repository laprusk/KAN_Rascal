#include "edge_mlp.h"
#include "kan.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>


// 活性化関数適用後の出力
//double aout[EMLP_NUM_LAYERS][EMLP_MAX_NODES];


void emlp_init(
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES]
) {
	int l;

	// ノード数配列を作成
	int num_nodes[EMLP_NUM_LAYERS] = { 1, EMLP_D, 1 };

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


double emlp_forward(
	double x,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
) {
	int l;

	// 入力を0層の出力へコピー
	out[0][0] = x;

	// ノード数配列を作成
	int num_nodes[EMLP_NUM_LAYERS] = { 1, EMLP_D, 1 };

	// forward
	for (l = 1; l < EMLP_NUM_LAYERS; ++l) {
		for (int j = 0; j < num_nodes[l]; ++j) {
			out[l][j] = bias[l - 1][j];
			for (int i = 0; i < num_nodes[l - 1]; ++i) {
				out[l][j] += weight[l - 1][j][i] * out[l - 1][i];
			}
			
			// activation
			if (l != EMLP_NUM_LAYERS - 1) out[l][j] = relu(out[l][j]);
		}
	}

	return out[EMLP_NUM_LAYERS - 1][0];
}


double emlp_backprop(
	double kan_delta,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double delta[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
) {

	int l;

	// 最終層のデルタ
	delta[EMLP_NUM_LAYERS - 1][0] = kan_delta;

	// ノード数配列を作成
	int num_nodes[EMLP_NUM_LAYERS] = { 1, EMLP_D, 1 };

	// backward
	for (l = EMLP_NUM_LAYERS - 2; l >= 0; --l) {
		for (int i = 0; i < num_nodes[l]; ++i) {
			delta[l][i] = 0;
			if (out[l][i] != 0 || l == 0) {			// 出力が0なら、活性化関数によらず誤差信号も0
				for (int j = 0; j < num_nodes[l + 1]; ++j) {
					delta[l][i] += delta[l + 1][j] * weight[l][j][i];
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

	return delta[0][0];
}

void emlpk_init(
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES]
) {
	int l;

	// ノード数配列を作成
	int num_nodes[EMLP_NUM_LAYERS] = { EMLP_K, EMLP_D, 1 };

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

double emlpk_forward(
	double x,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
) {
	int l;

	// 入力を0層の出力へコピー
	out[0][0] = x;

	// ノード数配列を作成
	int num_nodes[EMLP_NUM_LAYERS] = { EMLP_K, EMLP_D, 1 };

	// 入力をK乗まで拡張する
	for (int i = 1; i < EMLP_K; ++i) {
		out[0][i] = powf(x, i + 1);
	}

	// forward
	for (l = 1; l < EMLP_NUM_LAYERS; ++l) {
		for (int j = 0; j < num_nodes[l]; ++j) {
			out[l][j] = bias[l - 1][j];
			for (int i = 0; i < num_nodes[l - 1]; ++i) {
				out[l][j] += weight[l - 1][j][i] * out[l - 1][i];
			}

			// activation
			if (l != EMLP_NUM_LAYERS - 1) out[l][j] = relu(out[l][j]);
		}
	}

	return out[EMLP_NUM_LAYERS - 1][0] + x;
}


double emlpk_backprop(
	double kan_delta,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double delta[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
) {

	int l;

	// 最終層のデルタ
	delta[EMLP_NUM_LAYERS - 1][0] = kan_delta;

	// ノード数配列を作成
	int num_nodes[EMLP_NUM_LAYERS] = { EMLP_K, EMLP_D, 1 };

	// backward
	for (l = EMLP_NUM_LAYERS - 2; l >= 0; --l) {
		for (int i = 0; i < num_nodes[l]; ++i) {
			delta[l][i] = 0;
			if (out[l][i] != 0 || l == 0) {			// 出力が0なら、活性化関数によらず誤差信号も0
				for (int j = 0; j < num_nodes[l + 1]; ++j) {
					delta[l][i] += delta[l + 1][j] * weight[l][j][i];
				}
			}
		}
	}

	// 入力拡張のbackward
	double ret_delta = 0;
	for (int i = 0; i < EMLP_K; ++i) {
		ret_delta += pow(out[0][0], i) * delta[0][i];
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

	return ret_delta + kan_delta;
}

void emlps_init(
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES]
) {
	int l;

	// ノード数配列を作成
	int num_nodes[EMLP_NUM_LAYERS] = { 3, EMLP_D, 1 };

	// hidden layer
	for (l = 0; l < EMLP_NUM_LAYERS - 2; ++l) {
		// He初期化
		double num_nodes_cur = num_nodes[l];
		if (l == 0) num_nodes_cur = 1;
		double he_wp = 2 * sqrt(6.0 / num_nodes_cur);
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

double emlps_forward(
	double x,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
) {
	int l;

	// ノード数配列を作成
	int num_nodes[EMLP_NUM_LAYERS] = { 3, EMLP_D, 1 };

	// 入力の値に応じてノードを活性化
	for (int i = 0; i < 3; ++i) {
		out[0][i] = 0;
	}
	if (x < -EMLPS_BOUND) out[0][0] = x;
	else if (x < 0) out[0][1] = x;
	else if (x < EMLPS_BOUND) out[0][2] = x;
	else out[0][2] = x;

	// forward
	for (l = 1; l < EMLP_NUM_LAYERS; ++l) {
		for (int j = 0; j < num_nodes[l]; ++j) {
			out[l][j] = bias[l - 1][j];
			for (int i = 0; i < num_nodes[l - 1]; ++i) {
				out[l][j] += weight[l - 1][j][i] * out[l - 1][i];
			}

			// activation
			if (l != EMLP_NUM_LAYERS - 1) out[l][j] = relu(out[l][j]);
		}
	}

	return out[EMLP_NUM_LAYERS - 1][0];
}


double emlps_backprop(
	double kan_delta,
	double weight[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES][EMLP_MAX_NODES],
	double bias[EMLP_NUM_LAYERS - 1][EMLP_MAX_NODES],
	double out[EMLP_NUM_LAYERS][EMLP_MAX_NODES],
	double delta[EMLP_NUM_LAYERS][EMLP_MAX_NODES]
) {

	int l;

	// 最終層のデルタ
	delta[EMLP_NUM_LAYERS - 1][0] = kan_delta;

	// ノード数配列を作成
	int num_nodes[EMLP_NUM_LAYERS] = { 3, EMLP_D, 1 };

	// backward
	for (l = EMLP_NUM_LAYERS - 2; l >= 0; --l) {
		for (int i = 0; i < num_nodes[l]; ++i) {
			delta[l][i] = 0;
			if (out[l][i] != 0 || l == 0) {			// 出力が0なら、活性化関数によらず誤差信号も0
				for (int j = 0; j < num_nodes[l + 1]; ++j) {
					delta[l][i] += delta[l + 1][j] * weight[l][j][i];
				}
			}
		}
	}

	// スプライン活性のbackward
	double ret_delta = 0;
	for (int i = 0; i < 3; ++i) {
		ret_delta += delta[0][i];
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

	return ret_delta;
}
