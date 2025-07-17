#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "kan.h"
#include "kan_function.h"
#include "activation.h"
#include "util.h"


void kan_init(
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double coeff[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][NUM_CP],
	double knots[NUM_KNOTS]
) {

	if (!USE_ONLY_COEFFICIENT) {

		// wb��Xavier������
		for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
			double xavier_weight = sqrt(6.0 / (num_nodes[l] + num_nodes[l + 1]));
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				for (int i = 0; i < num_nodes[l]; ++i) {
					wb[l][j][i] = xavier_weight * (((double)rand() / RAND_MAX) * 2 - 1);
					//wb[l][j][i] = 0;
				}
			}
		}

		// ws��1�ŏ�����
		for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				for (int i = 0; i < num_nodes[l]; ++i) {
					ws[l][j][i] = 1;
				}
			}
		}

	}

	// coeff��N(0, 0.01)
	for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			for (int i = 0; i < num_nodes[l]; ++i) {
				for (int c = 0; c < NUM_CP; ++c) {
					coeff[l][j][i][c] = randn(0, 0.1);
				}
			}
		}
	}

	// �m�b�g�x�N�g��
	knots[0] = -(double)SPLINE_ORDER / GRID_SIZE;
	for (int k = 1; k < NUM_KNOTS; ++k) {
		knots[k] = knots[k - 1] + 1.0 / GRID_SIZE;
	}

}


void kan_forward(
	double x[DIM],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double coeff[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][NUM_CP],
	double knots[NUM_KNOTS],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES],
	double spline_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double basis_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	KANFunction func_type
) {

	// ���͂�0�w�̏o�͂փR�s�[
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
				spline_out[l][j][i] = spline(out[l][i], coeff[l][j][i], knots, basis_out[l][i], func_type);

				if (USE_ONLY_COEFFICIENT) out[l + 1][j] += spline_out[l][j][i];
				else out[l + 1][j] += wb[l][j][i] * silu_out[l][i] + ws[l][j][i] * spline_out[l][j][i];
			}
		}
	}

	// �ŏI�w��softmax
	const int last_layer = KAN_NUM_LAYERS - 1;
	softmax(out[last_layer], num_nodes[last_layer]);

}


void kan_backprop(
	bool t[NUM_CLASSES],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double coeff[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][NUM_CP],
	double knots[NUM_KNOTS],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double delta[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES],
	double spline_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double basis_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	KANFunction func_type
) {

	const int last_layer = KAN_NUM_LAYERS - 1;

	// output layer (squared err or cross-entropy err)
	for (int i = 0; i < NUM_CLASSES; ++i) {
		delta[last_layer][i] = out[last_layer][i] - (double)t[i];
	}

	// hidden layer
	for (int l = KAN_NUM_LAYERS - 2; l > 0; --l) {
		for (int i = 0; i < num_nodes[l]; ++i) {
			delta[l][i] = 0;
			const double sig_out = sigmoid(out[l][i]);
			const double dsilu = sig_out + out[l][i] * sig_out * (1 - sig_out);
			//const double dsilu = 1;
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				const double dspline = spline_derive(out[l][i], coeff[l][j][i], knots, basis_out[l][i], func_type);
				//const double dspline = 1;

				if (USE_ONLY_COEFFICIENT) delta[l][i] += dspline * delta[l + 1][j];
				delta[l][i] += (wb[l][j][i] * dsilu + ws[l][j][i] * dspline) * delta[l + 1][j];
			}
		}
	}

	// update spline coefficient
	for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			for (int i = 0; i < num_nodes[l]; ++i) {
				for (int c = 0; c < NUM_CP; ++c) {
					if (USE_ONLY_COEFFICIENT) coeff[l][j][i][c] -= KAN_LR * (delta[l + 1][j] * basis_out[l][i][c]);
					else coeff[l][j][i][c] -= KAN_LR * (delta[l + 1][j] * ws[l][j][i] * basis_out[l][i][c]);
				}
			}
		}
	}

	if (!USE_ONLY_COEFFICIENT) {
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
