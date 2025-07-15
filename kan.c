#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "kan.h"
#include "activation.h"
#include "util.h"


void kan_init(
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES],
	double coeff[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][KAN_MAX_NODES][NUM_CP],
	double knots[NUM_KNOTS]
) {

	// wbはXavier初期化
	for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			for (int i = 0; i < num_nodes[l]; ++i) {
				wb[l][j][i] = sqrt(6.0 / (num_nodes[l] + num_nodes[l + 1])) * (((double)rand() / RAND_MAX) * 2 - 1);
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
	knots[0] = -(double)SPLINE_ORDER / GRID_SIZE;
	for (int k = 1; k < NUM_KNOTS; ++k) {
		knots[k] = knots[k - 1] + 1.0 / GRID_SIZE;
	}

}


// de Boor Cox漸化式
double de_boor_cox(double x, int i, int order, double knots[NUM_KNOTS]) {

	if (order == 0) {
		if (knots[i] <= x && x <= knots[i + 1]) return 1;
		else return 0;
	}

	const double res =
		(x - knots[i]) / (knots[i + order] - knots[i]) * de_boor_cox(x, i, order - 1, knots) +
		(knots[i + order + 1] - x) / (knots[i + order + 1] - knots[i + 1]) * de_boor_cox(x, i + 1, order - 1, knots);

	return res;
}


// Bスプライン曲線
double bspline(double x, double coeff[NUM_CP], double knots[NUM_KNOTS], double basis_out[NUM_CP]) {
	double sum = 0;

	for (int i = 0; i < NUM_CP; ++i) {
		basis_out[i] = de_boor_cox(x, i, SPLINE_ORDER, knots);
		sum += coeff[i] * basis_out[i];
	}

	return sum;
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
	double basis_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP]
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
				spline_out[l][j][i] = bspline(out[l][i], coeff[l][j][i], knots, basis_out[l][i]);

				out[l + 1][j] += wb[l][j][i] * silu_out[l][i] + ws[l][j][i] * spline_out[l][j][i];
			}
		}
	}

	softmax(out[KAN_NUM_LAYERS - 1], num_nodes[KAN_NUM_LAYERS - 1]);

}


// de Boor Cox漸化式の微分
double de_boor_cox_derive(double x, int i, int order, double knots[NUM_KNOTS]) {
	const double res =
		order / (knots[i + order] - knots[i]) * de_boor_cox(x, i, order - 1, knots) -
		order / (knots[i + order + 1] - knots[i + 1]) * de_boor_cox(x, i + 1, order - 1, knots);

	return res;
}


// Bスプライン曲線の微分
double bspline_derive(double x, double coeff[NUM_CP], double knots[NUM_KNOTS], double basis_out[NUM_CP]) {
	double sum = 0;

	for (int i = 0; i < NUM_CP; ++i) {
		sum += coeff[i] * basis_out[i];
	}

	return sum;
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
	double basis_out[KAN_NUM_LAYERS - 1][KAN_MAX_NODES][NUM_CP]
) {

	const int last_layer = KAN_NUM_LAYERS - 1;

	// output layer (squared err)
	for (int i = 0; i < NUM_CLASSES; ++i) {
		delta[last_layer][i] = out[last_layer][i] - (double)t[i];
	}

	// hidden layer
	for (int l = KAN_NUM_LAYERS - 2; l > 0; --l) {
		for (int i = 0; i < num_nodes[l]; ++i) {
			delta[l][i] = 0;
			for (int j = 0; j < num_nodes[l + 1]; ++j) {
				const double sig_out = sigmoid(out[l][i]);
				const double dsilu = sig_out * out[l][i] * sig_out * (1 - sig_out);
				const double dspline = bspline_derive(out[l][i], coeff[l][j][i], knots, basis_out[l][i]);
				//double const dsilu = 1;
				//double const dspline = 1;

				delta[l][i] += (wb[l][j][i] * dsilu + ws[l][j][i] * dspline) * delta[l + 1][j];
			}
		}
	}

	// update wb, ws
	for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			for (int i = 0; i < num_nodes[l]; ++i) {
				wb[l][j][i] -= KAN_LR * (delta[l + 1][j] * silu_out[l][i]);
				ws[l][j][i] -= KAN_LR * (delta[l + 1][j] * spline_out[l][j][i]);
			}
		}
	}

	// update spline coefficient
	for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			for (int i = 0; i < num_nodes[l]; ++i) {
				for (int c = 0; c < NUM_CP; ++c) {
					coeff[l][j][i][c] -= KAN_LR * (delta[l + 1][j] * ws[l][j][i] * basis_out[l][i][c]);
				}
			}
		}
	}

}
