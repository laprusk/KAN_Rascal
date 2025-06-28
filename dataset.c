#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>


void load_dataset(
	double train_data[][DIM],
	double test_data[][DIM],
	int train_label[],
	int test_label[]
) {
	FILE* fp;
	errno_t r;

	// Training data
	r = fopen_s(&fp, train_file_name, "r");
	if (r != 0) return;
	for (int i = 0; i < NUM_TRAINS; ++i) {
		for (int j = 0; j < DIM; ++j) {
			int temp;
			fscanf_s(fp, "%d, ", &temp);
			train_data[i][j] = temp / MAXV;
		}
		fscanf_s(fp, "%d\n", &train_label[i]);
	}
	fclose(fp);

	// Test data
	r = fopen_s(&fp, test_file_name, "r");
	if (r != 0) return;
	for (int i = 0; i < NUM_TESTS; ++i) {
		for (int j = 0; j < DIM; ++j) {
			int temp;
			fscanf_s(fp, "%d, ", &temp);
			test_data[i][j] = temp / MAXV;
		}
		fscanf_s(fp, "%d\n", &test_label[i]);
	}
	fclose(fp);
}


void convert_one_hot(int label, bool* tk) {
	for (int i = 0; i < NUM_CLASSES; ++i) {
		tk[i] = (i == label) ? 1 : 0;
	}
}

