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
	double knots[NUM_KNOTS],
	double phase_low[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	double phase_height[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	KANFunction func_type
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

	// coeffはN(0, 0.01)
	for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			for (int i = 0; i < num_nodes[l]; ++i) {
				for (int c = 0; c < NUM_CP; ++c) {
					coeff[l][j][i][c] = randn(0, 0.1);
				}
			}
		}
	}

	// ノットベクトル
	if (func_type == B_SPLINE) {
		const double grid_step = (double)(GRID_MAX - GRID_MIN) / GRID_SIZE;
		knots[0] = GRID_MIN - grid_step * SPLINE_ORDER;
		for (int i = 1; i < NUM_KNOTS; ++i) {
			knots[i] = knots[i - 1] + grid_step;
		}
	}
	else {
		const double grid_step = (double)(GRID_MAX - GRID_MIN) / NUM_CP;
		knots[0] = GRID_MIN;
		for (int i = 1; i < NUM_CP; ++i) {
			knots[i] = knots[i - 1] + grid_step;
		}
	}

	// ReLU-KAN
	if (func_type == RELU_KAN) {
		const double grid_step = (double)(GRID_MAX - GRID_MIN) / GRID_SIZE;

		for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
			for (int i = 0; i < num_nodes[l]; ++i) {
				phase_low[l][i][0] = GRID_MIN - grid_step * SPLINE_ORDER;
				phase_height[l][i][0] = GRID_MIN + grid_step;
				for (int c = 1; c < NUM_CP; ++c) {
					phase_low[l][i][c] = phase_low[l][i][c - 1] + grid_step;
					phase_height[l][i][c] = phase_height[l][i][c - 1] + grid_step;
				}
			}
		}
	}

}


void kan_forward(
	double x[DIM],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double coeff[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][NUM_CP],
	double knots[NUM_KNOTS],
	double phase_low[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	double phase_height[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES],
	double silu_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES],
	double spline_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double basis_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	KANFunction func_type
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
				spline_out[l][j][i] = spline(out[l][i], coeff[l][j][i], knots, phase_height, phase_low, basis_out[l][i], func_type);

				if (NO_WEIGHT_AND_BASIS) out[l + 1][j] += spline_out[l][j][i];
				else out[l + 1][j] += wb[l][j][i] * silu_out[l][i] + ws[l][j][i] * spline_out[l][j][i];
			}
		}
	}

	// 最終層でsoftmax
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
	double phase_low[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
	double phase_height[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP],
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
				const double dspline = spline_derive(out[l][i], coeff[l][j][i], knots, phase_height, phase_low, basis_out[l][i], func_type);
				//const double dspline = 1;

				if (NO_WEIGHT_AND_BASIS) delta[l][i] += dspline * delta[l + 1][j];
				delta[l][i] += (wb[l][j][i] * dsilu + ws[l][j][i] * dspline) * delta[l + 1][j];
			}
		}
	}

	// update spline coefficient
	for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			for (int i = 0; i < num_nodes[l]; ++i) {
				for (int c = 0; c < NUM_CP; ++c) {
					if (NO_WEIGHT_AND_BASIS) coeff[l][j][i][c] -= KAN_LR * (delta[l + 1][j] * basis_out[l][i][c]);
					else coeff[l][j][i][c] -= KAN_LR * (delta[l + 1][j] * ws[l][j][i] * basis_out[l][i][c]);
				}
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
