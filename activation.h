#pragma once

typedef enum {
	SIGMOID,
	RELU,
	SILU,
	GELU,
	SOFTMAX
} Activation;

double sigmoid(double x);
double relu(double x);
double relu_derive(double x);
double silu(double x);
void softmax(double* x, int n);
