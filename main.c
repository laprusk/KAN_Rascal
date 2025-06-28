#include "mlp.h"
#include "dataset.h"
#include "util.h"
#include <stdio.h>
#include <stdbool.h>


const int MLP_EPOCH_MAX = 10;


DataSet dataset;

MLP model = {
		2,
		{DIM, DIM, CLASSES},
		0.01,
		RELU,
		SOFTMAX
};


void train_mlp() {

	bool tk[CLASSES];
	int train_order[NUM_TRAINS];

	mlp_initialize(&model);

	for (int ep = 0; ep < MLP_EPOCH_MAX; ++ep) {
		// shuffle dataset order
		for (int i = 0; i < NUM_TRAINS; ++i) train_order[i] = i;
		shuffle(train_order, NUM_TRAINS);

		// train
		for (int t = 0; t < NUM_TRAINS; ++t) {
			int i = train_order[t];
			convert_one_hot(dataset.train_label[i], tk, CLASSES);

			mlp_forward(&model, dataset.train_data[i], DIM);
			mlp_backprop(&model, tk, CLASSES);
		}

		// test
		int count = 0;
		for (int t = 0; t < NUM_TESTS; ++t) {
			int i = t;
			mlp_forward(&model, dataset.test_data[i], DIM);
			if (mlp_is_collect(&model, dataset.test_label[i])) ++count;
		}
		printf("Epoch: %d, %.3f\n", ep, (double)count / NUM_TESTS);
	}

}


int main() {
	
	// データセット読み込み
	load_dataset(&dataset);

	train_mlp();
}
