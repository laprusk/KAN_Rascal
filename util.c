#include "util.h"
#include <stdlib.h>


void shuffle(int* arr, int n) {
	for (int i = 0; i < n - 1; ++i) {
		int j = rand() / (RAND_MAX + 1.0) * (n - i);
		int temp = arr[i];
		arr[i] = arr[i + j];
		arr[i + j] = temp;
	}
}
