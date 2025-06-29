#include "mlp.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>


// �萔�̍Ē�`
const int NUM_LAYERS = MLP_NUM_LAYERS;


/// <summary>
/// MLP�̏d�ݏ�����
/// ReLU�̏ꍇ��He���������s���A����ȊO�ł͈�l���������s��
/// </summary>
/// <param name="model">MLP���f��</param>
void mlp_init(
	int num_nodes[MLP_NUM_LAYERS],
	double weight[MLP_NUM_LAYERS - 1][MLP_MAX_NODES][MLP_MAX_NODES],
	double bias[MLP_NUM_LAYERS - 1][MLP_MAX_NODES],
	Activation hidden_activation
) {
	int l;

	// hidden layer
	for (l = 0; l < NUM_LAYERS - 2; ++l) {
		// ReLU�Ȃ�He������
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
		// ReLU�ȊO�Ȃ��l������
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
/// MLP���`�d
/// </summary>
/// <param name="model">MLP���f��</param>
/// <param name="input">����</param>
/// <param name="dim_input">���͎���</param>
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

	// ���͂�0�w�̏o�͂փR�s�[
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
/// MLP�t�`�d
/// </summary>
/// <param name="model">MLP���f��</param>
/// <param name="tk">���t�f�[�^</param>
/// <param name="num_classes">�N���X��</param>
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
			if (out[l][i] != 0) {			// �o�͂�0�Ȃ�A�������֐��ɂ�炸�덷�M����0
				for (int j = 0; j < num_nodes[l + 1]; ++j)
					delta[l][i] += delta[l + 1][j] * weight[l][j][i];

				if (hidden_activation == SIGMOID) delta[l][i] *= (1 - out[l][i]) * out[l][i];
			}
		}
	}
	if (is_delta_0layer) {
		for (int i = 0; i < num_nodes[l]; ++i) {
			delta[l][i] = 0;
			if (out[l][i] != 0) {			// �o�͂�0�Ȃ�A�������֐��ɂ�炸�덷�M����0
				for (int j = 0; j < num_nodes[l + 1]; ++j)
					delta[l][i] += delta[l + 1][j] * weight[l][j][i];

				if (hidden_activation == SIGMOID) delta[l][i] *= (1 - out[l][i]) * out[l][i];
			}
		}
	}

	// �d�݁E�o�C�A�X�̍X�V
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
/// ���_���ʂ����������ǂ������ׂ�
/// </summary>
/// <param name="model">MLP���f��</param>
/// <param name="label">���t���x��</param>
/// <returns>���_���ʂ����������true�A�ԈႦ�Ă����false</returns>
bool mlp_is_collect(double output[], int label) {

	int maxi = 0;
	for (int i = 1; i < NUM_CLASSES; ++i) {
		if (output[i] > output[maxi]) maxi = i;
	}

	return maxi == label;
}

