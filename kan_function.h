#pragma once

#include "kan.h"

#define GRID_MIN (-(double)SPLINE_ORDER / GRID_SIZE)
#define GRID_MAX (1.0 + ((double)SPLINE_ORDER / GRID_SIZE))
#define DENOMINATOR ((GRID_MAX - GRID_MIN) / NUM_KNOTS)

double spline(double x, double coeff[NUM_CP], double knots[NUM_KNOTS], double basis_out[NUM_CP], KANFunction func_type);
double spline_derive(double x, double coeff[NUM_CP], double knots[NUM_KNOTS], double basis_out[NUM_CP], KANFunction func_type);
