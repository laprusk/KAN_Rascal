#pragma once

#include "kan.h"

static const double INV_DENOMINATOR = 1 / ((GRID_MAX - GRID_MIN) / NUM_KNOTS);

double spline(
	double x, double coeff[NUM_CP], double knots[NUM_KNOTS], double phase_low[NUM_CP], double phase_height[NUM_CP],
	double basis_out[NUM_CP], KANFunction func_type
);
double spline_derive(
	double x, double coeff[NUM_CP], double knots[NUM_KNOTS], double phase_low[NUM_CP], double phase_height[NUM_CP],
	double basis_out[NUM_CP], KANFunction func_type
);
