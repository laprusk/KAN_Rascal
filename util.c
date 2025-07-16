#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <math.h>
#include "util.h"

void shuffle(int* arr, int n) {
	for (int i = 0; i < n - 1; ++i) {
		int j = rand() / (RAND_MAX + 1.0) * (n - i);
		int temp = arr[i];
		arr[i] = arr[i + j];
		arr[i + j] = temp;
	}
}


double randn(double mu, double sigma) {
	double u1, u2;

	u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
	u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

	return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2) * sigma + mu;
}
