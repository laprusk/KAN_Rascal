#include "kan.h"
#include "activation.h"


// de Boor Cox漸化式
double de_boor_cox(double x, int i, int order, double knots[NUM_KNOTS]) {

	if (order == 0) {
		if (knots[i] <= x <= knots[i + 1]) return 1;
		else return 0;
	}

	double res =
		(x - knots[i]) / (knots[i + order] - knots[i]) * de_boor_cox(x, i, order - 1, knots) +
		(knots[i + order + 1] - x) / (knots[i + order + 1] - knots[i + 1]) * de_boor_cox(x, i + 1, order - 1, knots);

	return res;
}


// Bスプライン曲線
double bspline(double x, double coeff[NUM_CP], double knots[NUM_KNOTS]) {
	double sum = 0;

	for (int i = 0; i < NUM_CP; ++i) {
		sum += coeff[i] * de_boor_cox(x, i, SPLINE_ORDER, knots);
	}
}


void kan_forward(
	double x[DIM],
	int num_nodes[KAN_NUM_LAYERS],
	double wb[KAN_NUM_LAYERS][KAN_MAX_NODES][KAN_MAX_NODES],
	double ws[KAN_NUM_LAYERS][KAN_MAX_NODES][KAN_MAX_NODES],
	double coeff[KAN_NUM_LAYERS][KAN_MAX_NODES][KAN_MAX_NODES][NUM_CP],
	double knots[NUM_KNOTS],
	double out[KAN_NUM_LAYERS][KAN_MAX_NODES]
) {

	// 入力を0層の出力へコピー
	for (int i = 0; i < DIM; ++i) {
		out[0][i] = x[i];
	}

	// 各エッジのBスプラインに通した結果を、各ノードで和を取っていく
	for (int l = 0; l < KAN_NUM_LAYERS - 1; ++l) {
		for (int j = 0; j < num_nodes[l + 1]; ++j) {
			out[l + 1][j] = 0;
			for (int i = 0; i < num_nodes[l]; ++i) {
				out[l + 1][j] +=
					wb[l][j][i] * silu(out[l][i]) +
					ws[l][j][i] * bspline(out[l][i], coeff[l][j][i], knots);
			}
		}
	}

	// softmax
	const int last_layer = KAN_NUM_LAYERS - 1;
	softmax(out[last_layer], num_nodes[last_layer]);

}
