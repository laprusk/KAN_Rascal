#pragma once

#define KAN_NUM_LAYERS 3
#define KAN_INPUT_DIM DIM
#define KAN_MAX_NODES DIM
#define LR 0.01
#define GRID_SIZE 5
#define SPLINE_ORDER 3
#define NUM_CP (GRID_SIZE + SPLINE_ORDER)
#define NUM_KNOTS (GRID_SIZE + 1 + SPLINE_ORDER * 2)
