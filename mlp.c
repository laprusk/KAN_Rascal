#include "mlp.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>


// 定数の再定義
const int NUM_LAYERS = MLP_NUM_LAYERS;


/// <summary>
/// MLPの重み初期化
/// ReLUの場合はHe初期化を行い、それ以外では一様初期化を行う
/// </summary>
/// <param name="model">MLPモデル</param>
void mlp_init(
	int num_nodes[MLP_NUM_LAYERS],
	double weight[MLP_NUM_LAYERS - 1][MLP_MAX_NODES][MLP_MAX_NODES],
	double bias[MLP_NUM_LAYERS - 1][MLP_MAX_NODES],
	Activation hidden_activation
) {
	int l;

	// hidden layer
	for (l = 0; l < NUM_LAYERS - 2; ++l) {
		// ReLUならHe初期化
		if (hidden_activation == RELU) {
			double he_wp = 2 * sqrt(6.0 / num_nodes[l]);
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				bias[l][j] = 0;
				for (int i = 0; i < num_nodes[l]; ++i) {
					weight[l][j][i] = ((double)rand() / RAND_MAX - 0.5) * he_wp;
					//woi[l][j][i] = randn() / sqrt(NodeN[l] / 2);
				}
			}
		}
		// ReLU以外なら一様初期化
		else {
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				bias[l][j] = 0;
				for (int i = 0; i < num_nodes[l]; ++i) {
					weight[l][j][i] = (double)rand() / (RAND_MAX)-0.5;
				}
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


/// <summary>
/// MLP順伝播
/// </summary>
/// <param name="model">MLPモデル</param>
/// <param name="input">入力</param>
/// <param name="dim_input">入力次元</param>
void mlp_forward(
	double x[DIM],
	int num_nodes[MLP_NUM_LAYERS],
	double weight[MLP_NUM_LAYERS - 1][MLP_MAX_NODES][MLP_MAX_NODES],
	double bias[MLP_NUM_LAYERS - 1][MLP_MAX_NODES],
	double out[MLP_NUM_LAYERS][MLP_MAX_NODES],
	Activation hidden_activation,
	Activation output_activation
) {
	int l;

	// 入力を0層の出力へコピー
	for (int i = 0; i < DIM; ++i) {
		out[0][i] = x[i];
	}

	// hidden layer
	for (l = 1; l < NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l]; ++j) {
			out[l][j] = bias[l - 1][j];
			for (int i = 0; i < num_nodes[l - 1]; ++i) {
				out[l][j] += weight[l - 1][j][i] * out[l - 1][i];
			}

			if (hidden_activation == RELU) out[l][j] = relu(out[l][j]);
			else if (hidden_activation == SIGMOID) out[l][j] = sigmoid(out[l][j]);
		}
	}

	// output layer
	for (int j = 0; j < num_nodes[l]; ++j) {
		out[l][j] = bias[l - 1][j];
		for (int i = 0; i < num_nodes[l - 1]; ++i) {
			out[l][j] += weight[l - 1][j][i] * out[l - 1][i];
		}
	}
	if (output_activation == SOFTMAX) {
		softmax(out[l], num_nodes[l]);
	}
	else if (output_activation == SIGMOID) {
		for (int j = 0; j < num_nodes[l]; ++j) {
			out[l][j] = sigmoid(out[l][j]);
		}
	}

}


/// <summary>
/// MLP逆伝播
/// </summary>
/// <param name="model">MLPモデル</param>
/// <param name="tk">教師データ</param>
/// <param name="num_classes">クラス数</param>
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
) {

	int l;

	// output layer
	l = NUM_LAYERS - 1;
	// Softmax with cross entropy error
	if (output_activation == SOFTMAX) {
		for (int i = 0; i < NUM_CLASSES; ++i) {
			delta[l][i] = out[l][i] - (double)t[i];
		}
	}
	// Sigmoid with squared error
	else if (output_activation == SIGMOID) {
		for (int i = 0; i < NUM_CLASSES; ++i) {
			delta[l][i] = (out[l][i] - (double)t[i]) * (1 - out[l][i]) * out[l][i];
		}
	}

	// hidden layer
	for (l = NUM_LAYERS - 2; l > 0; --l) {
		for (int i = 0; i < num_nodes[l]; ++i) {
			delta[l][i] = 0;
			if (out[l][i] != 0) {			// 出力が0なら、活性化関数によらず誤差信号も0
				for (int j = 0; j < num_nodes[l + 1]; ++j)
					delta[l][i] += delta[l + 1][j] * weight[l][j][i];

				if (hidden_activation == SIGMOID) delta[l][i] *= (1 - out[l][i]) * out[l][i];
			}
		}
	}
	if (is_delta_0layer) {
		for (int i = 0; i < num_nodes[l]; ++i) {
			delta[l][i] = 0;
			if (out[l][i] != 0) {			// 出力が0なら、活性化関数によらず誤差信号も0
				for (int j = 0; j < num_nodes[l + 1]; ++j)
					delta[l][i] += delta[l + 1][j] * weight[l][j][i];

				if (hidden_activation == SIGMOID) delta[l][i] *= (1 - out[l][i]) * out[l][i];
			}
		}
	}

	// 重み・バイアスの更新
	for (l = 0; l < NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			bias[l][j] -= LR * delta[l + 1][j];
			for (int i = 0; i < num_nodes[l]; ++i) {
				weight[l][j][i] -= LR * delta[l + 1][j] * out[l][i];
			}
		}
	}

}


/// <summary>
/// 推論結果が正しいかどうか調べる
/// </summary>
/// <param name="model">MLPモデル</param>
/// <param name="label">教師ラベル</param>
/// <returns>推論結果が正しければtrue、間違えていればfalse</returns>
bool mlp_is_collect(double output[], int label) {

	int maxi = 0;
	for (int i = 1; i < NUM_CLASSES; ++i) {
		if (output[i] > output[maxi]) maxi = i;
	}

	return maxi == label;
}

