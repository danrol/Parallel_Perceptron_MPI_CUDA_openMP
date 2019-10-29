#pragma once

#ifndef FILEMETHODS_H
#define FILEMETHODS_H

void throwFileError(char *errorMessage);
void numReadCheck(char *str, double num);
double readNumFromFile(char separator[], char *dataType);
void readInputFromFile(int *numOfPoints, int *numOfDimensions,
	double *alphaZero, double *alphaMax, int* limitIter,
	double *qc, double **points, int **pointGroups);
void outputResult(double alpha, double q, double *w, double numOfW);
void outputAlphaNotFound();

#endif