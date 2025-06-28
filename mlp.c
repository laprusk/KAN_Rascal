#include "mlp.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>


/// <summary>
/// MLPの重み初期化
/// ReLUの場合はHe初期化を行い、それ以外では一様初期化を行う
/// </summary>
/// <param name="model">MLPモデル</param>
void mlp_initialize(MLP* model) {
	int l;

	if (model->num_layers > MLP_MAX_LAYERS) return;

	// hidden layer
	for (l = 0; l < model->num_layers - 1; ++l) {
		// ReLUならHe初期化
		if (model->hidden_activation == RELU) {
			double he_wp = 2 * sqrt(6.0 / model->num_nodes[l]);
			for (int j = 0; j < model->num_nodes[l + 1]; ++j) {
				model->bias[l][j] = 0;
				for (int i = 0; i < model->num_nodes[l]; ++i) {
					model->weight[l][j][i] = ((double)rand() / RAND_MAX - 0.5) * he_wp;
					//woi[l][j][i] = randn() / sqrt(NodeN[l] / 2);
				}
			}
		}
		// ReLU以外なら一様初期化
		else {
			for (int j = 0; j < model->num_nodes[l + 1]; ++j) {
				model->bias[l][j] = 0;
				for (int i = 0; i < model->num_nodes[l]; ++i) {
					model->weight[l][j][i] = (double)rand() / (RAND_MAX)-0.5;
				}
			}
		}
	}

	// output layer
	for (int j = 0; j < model->num_nodes[l + 1]; ++j) {
		model->bias[l][j] = 0;
		for (int i = 0; i < model->num_nodes[l]; ++i) {
			model->weight[l][j][i] = (double)rand() / (RAND_MAX)-0.5;
		}
	}

}


/// <summary>
/// MLP順伝播
/// </summary>
/// <param name="model">MLPモデル</param>
/// <param name="input">入力</param>
/// <param name="dim_input">入力次元</param>
void mlp_forward(MLP* model, double* input, int dim_input) {
	int l;

	if (model->num_layers > MLP_MAX_LAYERS) return;

	// model->が冗長なため一度ローカルで再定義
	double (*weight)[MLP_MAX_NODES][MLP_MAX_NODES] = model->weight;
	double (*bias)[MLP_MAX_NODES] = model->bias;
	double (*out)[MLP_MAX_NODES] = model->out;

	// 入力を0層の出力へコピー
	for (int i = 0; i < dim_input; ++i) {
		out[0][i] = input[i];
	}

	// hidden layer
	for (l = 0; l < model->num_layers - 1; ++l) {
		for (int j = 0; j < model->num_nodes[l + 1]; ++j) {
			out[l + 1][j] = bias[l][j];
			for (int i = 0; i < model->num_nodes[l]; ++i) {
				out[l + 1][j] += weight[l][j][i] * out[l][i];
			}

			if (model->hidden_activation == RELU) out[l + 1][j] = relu(out[l + 1][j]);
			else if (model->hidden_activation == SIGMOID) out[l + 1][j] = sigmoid(out[l + 1][j]);
		}
	}

	// output layer
	for (int j = 0; j < model->num_nodes[l + 1]; ++j) {
		out[l + 1][j] = bias[l][j];
		for (int i = 0; i < model->num_nodes[l]; ++i) {
			out[l + 1][j] += weight[l][j][i] * out[l][i];
		}
	}
	if (model->out_activation == SOFTMAX) {
		softmax(out[l + 1], model->num_nodes[l + 1]);
	}
	else if (model->out_activation == SIGMOID) {
		for (int j = 0; j < model->num_nodes[l + 1]; ++j) {
			out[l + 1][j] = sigmoid(out[l + 1][j]);
		}
	}

}


/// <summary>
/// MLP逆伝播
/// </summary>
/// <param name="model">MLPモデル</param>
/// <param name="tk">教師データ</param>
/// <param name="num_classes">クラス数</param>
void mlp_backprop(MLP* model, bool* tk, int num_classes) {

	int l;

	if (model->num_layers > MLP_MAX_LAYERS) return;

	// model->が冗長なため一度ローカルで再定義
	double (*weight)[MLP_MAX_NODES][MLP_MAX_NODES] = model->weight;
	double (*bias)[MLP_MAX_NODES] = model->bias;
	double (*out)[MLP_MAX_NODES] = model->out;
	double (*delta)[MLP_MAX_NODES] = model->delta;

	// output layer
	l = model->num_layers;
	// Softmax with cross entropy error
	if (model->out_activation == SOFTMAX) {
		for (int i = 0; i < num_classes; ++i) {
			delta[l][i] = out[l][i] - (double)tk[i];
		}
	}
	// Sigmoid with squared error
	else if (model->out_activation == SIGMOID) {
		for (int i = 0; i < num_classes; ++i) {
			delta[l][i] = (out[l][i] - (double)tk[i]) * (1 - out[l][i]) * out[l][i];
		}
	}

	// hidden layer
	for (l = model->num_layers - 1; l > 0; --l) {
		for (int i = 0; i < model->num_nodes[l]; ++i) {
			delta[l][i] = 0;
			if (out[l][i] != 0) {			// 出力が0なら、活性化関数によらず誤差信号も0
				for (int j = 0; j < model->num_nodes[l + 1]; ++j)
					delta[l][i] += delta[l + 1][j] * weight[l][j][i];

				if (model->hidden_activation == SIGMOID) delta[l][i] *= (1 - out[l][i]) * out[l][i];
			}
		}
	}

	// 重み・バイアスの更新
	for (l = 0; l < model->num_layers; ++l) {
		for (int j = 0; j < model->num_nodes[l + 1]; ++j) {
			bias[l][j] -= model->learning_rate * delta[l + 1][j];
			for (int i = 0; i < model->num_nodes[l]; ++i) {
				weight[l][j][i] -= model->learning_rate * delta[l + 1][j] * out[l][i];
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
bool mlp_is_collect(MLP* model, int label) {

	const int last_layer = model->num_layers;
	const int last_nodes = model->num_nodes[last_layer];

	int maxi = 0;
	for (int i = 1; i < last_nodes; ++i) {
		if (model->out[last_layer][i] > model->out[last_layer][maxi]) maxi = i;
	}

	return maxi == label;
}

