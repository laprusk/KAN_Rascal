#pragma once

typedef enum {
	SIGMOID,
	RELU,
	SILU,
	GELU,
	SOFTMAX
} Activation;

double sigmoid(double x);
double sigmoid_derive(double x);
double relu(double x);
double relu_derive(double x);
double silu(double x);
double silu_derive(double x);
void softmax(double* x, int n);
