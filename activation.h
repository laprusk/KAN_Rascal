#pragma once

typedef enum {
	SIGMOID,
	RELU,
	SOFTMAX
} Activation;

double sigmoid(double x);
double relu(double x);
void softmax(double* x, int n);
