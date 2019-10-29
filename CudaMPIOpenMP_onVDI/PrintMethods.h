#pragma once

#ifndef PRINTMETHODS_H
#define PRINTMETHODS_H

void printW(double *w, int numOfW);
void printResults(int numOfPoints, int numOfDimensions, int iterLimit, double alpha,
	double alphaMax, double q, double qc, double* w);
void printPointsData(double *points, int numOfPoints, int numOfDimensions, int *pointsGroups);
void printPointGroups(int *pointGroups, int numOfPoints);
#endif