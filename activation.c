#include "activation.h"
#include <math.h>


double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

double sigmoid_derive(double x) {
	double sigx = sigmoid(x);
	return sigx * (1 - sigx);
}

double relu(double x) {
	if (x <= 0) return 0;
	else return x;
}

double relu_derive(double x) {
	if (x <= 0) return 0;
	else return 1;
}

double silu(double x) {
	return x / (1 + exp(-x));
}

double silu_derive(double x) {
	double sigx = sigmoid(x);
	return sigx + x * sigx * (1 - sigx);
}

void softmax(double* x, int n) {

	// ç≈ëÂíl
	double max = x[0];
	for (int i = 1; i < n; ++i) {
		if (x[i] > max) max = x[i];
	}

	// ëçòa
	double sum = 0.0;
	for (int i = 0; i < n; ++i) {
		x[i] = exp(x[i] - max);
		sum += x[i];
	}

	for (int i = 0; i < n; ++i) {
		x[i] /= sum;
	}

}
