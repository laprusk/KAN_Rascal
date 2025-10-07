#pragma once

#include <stdbool.h>

#define OPTDIGIT 0
#define MNIST 1

#define DATASET OPTDIGIT
#define DATASET MNIST

#if DATASET == OPTDIGIT

#define HEIGHT 8
#define WIDTH 8
#define CHANNEL 1
#define DIM (HEIGHT * WIDTH * CHANNEL)
#define NUM_CLASSES 10
#define NUM_TRAINS 3823
#define NUM_TESTS 1797
static const double MAXV = 16.0;
static const char* train_file_name = "optdigits_tra.csv";
static const char* test_file_name = "optdigits_tes.csv";

#elif DATASET == MNIST

#define HEIGHT 28
#define WIDTH 28
#define CHANNEL 1
#define DIM (HEIGHT * WIDTH * CHANNEL)
#define NUM_CLASSES 10
//#define NUM_TRAINS 60000
//#define NUM_TESTS 10000
#define NUM_TRAINS 4000
#define NUM_TESTS 2000
static const double MAXV = 255.0;
static const char* train_file_name = "dataset_mnist_train.csv";
static const char* test_file_name = "dataset_mnist_test.csv";

#endif

typedef struct {
	double train_data[NUM_TRAINS][DIM];
	double test_data[NUM_TESTS][DIM];
	int train_label[NUM_TRAINS];
	int test_label[NUM_TESTS];
} DataSet;

void load_dataset(
	double train_data[][DIM],
	double test_data[][DIM],
	int train_label[],
	int test_label[]
);
void convert_one_hot(int label, bool* tk);

//void load_dataset(DataSet* dataset);
//void convert_one_hot(int label, bool* tk, int num_classes);
