#pragma once

#include <stdbool.h>

#define OPTDIGIT 0
#define MNIST 1

#define DATASET OPTDIGIT

#if DATASET == OPTDIGIT

#define HEIGHT 8
#define WIDTH 8
#define CHANNEL 1
#define DIM (HEIGHT * WIDTH * CHANNEL)
#define CLASSES 10
#define NUM_TRAINS 3823
#define NUM_TESTS 1797
static const double MAXV = 16.0;
static const char* train_file_name = "optdigits_tra.csv";
static const char* test_file_name = "optdigits_tes.csv";

#endif


typedef struct {
	double train_data[NUM_TRAINS][DIM];
	double test_data[NUM_TESTS][DIM];
	int train_label[NUM_TRAINS];
	int test_label[NUM_TESTS];
} DataSet;


void load_dataset(DataSet* dataset);
void convert_one_hot(int label, bool* tk, int num_classes);
