#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>


void load_dataset(DataSet* dataset) {
	FILE* fp;
	errno_t r;

	// Training data
	r = fopen_s(&fp, train_file_name, "r");
	if (r != 0) return;
	for (int i = 0; i < NUM_TRAINS; ++i) {
		for (int j = 0; j < DIM; ++j) {
			int temp;
			fscanf_s(fp, "%d, ", &temp);
			dataset->train_data[i][j] = temp / MAXV;
		}
		fscanf_s(fp, "%d\n", &dataset->train_label[i]);
	}
	fclose(fp);

	// Test data
	r = fopen_s(&fp, test_file_name, "r");
	if (r != 0) return;
	for (int i = 0; i < NUM_TESTS; ++i) {
		for (int j = 0; j < DIM; ++j) {
			int temp;
			fscanf_s(fp, "%d, ", &temp);
			dataset->test_data[i][j] = temp / MAXV;
		}
		fscanf_s(fp, "%d\n", &dataset->test_label[i]);
	}
	fclose(fp);
}


void convert_one_hot(int label, bool* tk, int num_classes) {
	for (int i = 0; i < num_classes; ++i) {
		tk[i] = (i == label) ? 1 : 0;
	}
}

