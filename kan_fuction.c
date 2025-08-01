#include <math.h>
#include "kan_function.h"
#include "activation.h"


// Bスプライン基底関数
double b_spline_basis(double x, int i, int order, double knots[NUM_KNOTS]) {

	if (order == 0) {
		if (knots[i] <= x && x < knots[i + 1]) return 1;
		else return 0;
	}

	const double bi =
		(x - knots[i]) / (knots[i + order] - knots[i]) * b_spline_basis(x, i, order - 1, knots) +
		(knots[i + order + 1] - x) / (knots[i + order + 1] - knots[i + 1]) * b_spline_basis(x, i + 1, order - 1, knots);

	return bi;
}


// 動的計画法で高速化したBスプライン基底関数
double b_spline_basis_dp(double x, int i, int order, double knots[NUM_KNOTS]) {

	double dp[SPLINE_ORDER + 1][SPLINE_ORDER + 1];

	for (int j = 0; j < order + 1; ++j) {
		if (knots[i + j] <= x && x < knots[i + j + 1]) dp[0][j] = 1;
		else dp[0][j] = 0;
	}

	for (int k = 1; k <= order; ++k) {
		for (int j = 0; j < order + 1 - k; ++j) {
			dp[k][j] =
				(x - knots[i + j]) / (knots[i + j + k] - knots[i + j]) * dp[k - 1][j] +
				(knots[i + j + k + 1] - x) / (knots[i + j + k + 1] - knots[i + j + 1]) * dp[k - 1][j + 1];
		}
	}

	return dp[order][0];
}


// Gaussian Radial Basis Function (Fast-KAN)
double grbf(double x, double knot) {

	const double bi = exp(-pow((x - knot) * INV_DENOMINATOR, 2));
	
	return bi;
}


// Reflectional SWitch Activation Function (Faster-KAN)
double rswaf(double x, double knot) {

	const double bi = 1 - pow(tanh((x - knot) * INV_DENOMINATOR), 2);

	return bi;
}


// ReLU-KAN
double relu_kan(double x, double phase_low, double phase_height) {

	const double a = relu(x - phase_low);
	const double b = relu(phase_height - x);
	const double inv_norm = 16 / pow(phase_height - phase_low, 4);
	const double ri = pow(a * b, 2) * inv_norm;

	return ri;
}


// スプライン曲線
double spline(
	double x, double coeff[NUM_CP], double knots[NUM_KNOTS], double phase_low[NUM_CP], double phase_height[NUM_CP],
	double basis_out[NUM_CP], KANFunction func_type
) {

	double sum = 0;

	for (int i = 0; i < NUM_CP; ++i) {
		basis_out[i] = 0;
		if (func_type == B_SPLINE) basis_out[i] = b_spline_basis_dp(x, i, SPLINE_ORDER, knots);
		else if (func_type == GRBF) basis_out[i] = grbf(x, knots[i]);
		else if (func_type == RSWAF) basis_out[i] = rswaf(x, knots[i]);
		else if (func_type == RELU_KAN) basis_out[i] = relu_kan(x, phase_low[i], phase_height[i]);

		sum += coeff[i] * basis_out[i];
	}

	return sum;
}


// Bスプライン基底関数の微分
double b_spline_basis_derive(double x, int i, int order, double knots[NUM_KNOTS]) {

	const double res =
		order / (knots[i + order] - knots[i]) * b_spline_basis_dp(x, i, order - 1, knots) -
		order / (knots[i + order + 1] - knots[i + 1]) * b_spline_basis_dp(x, i + 1, order - 1, knots);

	return res;
}


// Gaussian Radial Basis Function (Fast-KAN)
double grbf_derive(double x, double knot , double basis_out) {

	const double dbi = -2 * basis_out * ((x - knot) * (INV_DENOMINATOR * INV_DENOMINATOR));
	//const double dbi = -2 * grbf(x, knot) * ((x - knot) * (INV_DENOMINATOR * INV_DENOMINATOR));

	return dbi;
}


// Reflectional SWitch Activation Function (Faster-KAN)
double rswaf_derive(double x, double knot, double basis_out) {

	const double dbi = (-2 * INV_DENOMINATOR) * basis_out * tanh((x - knot) * INV_DENOMINATOR);
	//const double dbi = (-2 * INV_DENOMINATOR) * rswaf(x, knot) * tanh((x - knot) * INV_DENOMINATOR);

	return dbi;
}


// ReLU-KAN
double relu_kan_derive(double x, double phase_low, double phase_height) {

	const double relu_pl = relu(x - phase_low);
	const double relu_ph = relu(phase_height - x);
	const double d_relu_pl = relu_derive(x - phase_low);
	const double d_relu_ph = relu_derive(phase_height - x);

	const double dri = 2 * relu_ph * relu_pl * (-d_relu_ph * relu_pl + relu_ph * d_relu_pl);

	return dri;
}


// スプライン曲線の微分
double spline_derive(
	double x, double coeff[NUM_CP], double knots[NUM_KNOTS], double phase_low[NUM_CP], double phase_height[NUM_CP],
	double basis_out[NUM_CP], KANFunction func_type
) {

	double sum = 0;

	for (int i = 0; i < NUM_CP; ++i) {
		double dbasis = 0;
		if (func_type == B_SPLINE) dbasis = b_spline_basis_derive(x, i, SPLINE_ORDER, knots);
		else if (func_type == GRBF) dbasis = grbf_derive(x, knots[i], basis_out[i]);
		else if (func_type == RSWAF) dbasis = rswaf_derive(x, knots[i], basis_out[i]);
		else if (func_type == RELU_KAN) dbasis = relu_kan_derive(x, phase_low[i], phase_height[i]);

		sum += coeff[i] * dbasis;
	}

	return sum;
}
