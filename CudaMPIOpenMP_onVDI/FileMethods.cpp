#include "mainMethods.h"
#include "FileMethods.h"
#include <errno.h>

void throwFileError(char *errorMessage)
{
	printf("\n");
	printf(errorMessage);
	printf("\n");
	exit(EXIT_FAILURE);
}

void numReadCheck(char *str, double num)
{
	if (str[0] != '0' && num == 0)
		throwFileError("Number read error");
}

double readNumFromFile(char separator[], char *dataType)
{
	double tmp;
	char *str;

	str = strtok(NULL, separator);

	if (dataType == INTGR)
		tmp = atoi(str);
	else if (dataType == DBL)
		tmp = (double)atof(str);
	else
		throwFileError("wrong data type");

	return tmp;

}

void readInputFromFile(int *numOfPoints, int *numOfDimensions,
	double *alphaZero, double *alphaMax, int* limitIter,
	double *qc, double **points, int **pointGroups)
{
	int lineLength = 1000, i, j, group;
	double tmp;
	char separator[] = " ";
	char* line = (char*)malloc(lineLength);
	printf(INPUT_FILE_PATH);
	FILE *file = fopen(INPUT_FILE_PATH, "r");
	char *token;
	if (file == NULL)
		throwFileError(FILE_OPEN_ERROR);

	if (fgets(line, lineLength, file) != NULL)
	{
		token = strtok(line, separator);
		if (token == NULL)
			throwFileError("Token equals NULL");

		*numOfPoints = atoi(token);
		*numOfDimensions = (int)readNumFromFile(separator, INTGR);
		*alphaZero = readNumFromFile(separator, DBL);
		*alphaMax = readNumFromFile(separator, DBL);
		*limitIter = (int)readNumFromFile(separator,  INTGR);
		*qc = readNumFromFile(separator, DBL);
	}

	(*points) = (double*)malloc((*numOfDimensions)*(*numOfPoints) * sizeof(double));
	(*pointGroups) = (int*)malloc((*numOfPoints) * sizeof(int));
	//fill points data
	for (i = 0; i < (*numOfPoints) && fgets(line, lineLength, file) != NULL; i++)
	{
		if (line == NULL)
			throwFileError("Line equals NULL");
		token = strtok(line, separator);

		//(*points)[i].dimensions[0] 
		(*points)[i*(*numOfDimensions)] = atof(token);
		numReadCheck(token, (*points)[i*(*numOfDimensions)]);

		for (j = 1; j < (*numOfDimensions); j++)
		{
			tmp = readNumFromFile(separator, DBL);
			(*points)[i*(*numOfDimensions) + j] = tmp;
		}
		group = (int)readNumFromFile(separator, INTGR);
		if (group != GROUP1 && group != GROUP2)
			throwFileError("Wrong group");
		(*pointGroups)[i] = group;
	}
	fclose(file);
	free(line);
}


void outputResult(double alpha, double q, double *w, double numOfW)
{
	FILE *file;
	int i;
	file = fopen(OUTPUT_FILE_PATH, "w");

	if (file == NULL)
		throwFileError(FILE_OPEN_ERROR);

	fprintf(file, "Alpha minimum = %f   q = %f\n", alpha, q);
	for (i = 0; i < numOfW; i++)
		fprintf(file, "W[%d] = %1f\n", i, w[i]);

	fclose(file);
}

void outputAlphaNotFound()
{
	FILE *file;
	file = fopen(OUTPUT_FILE_PATH, "w");

	if (file == NULL)
	{
		throwFileError(FILE_OPEN_ERROR);
	}
	fprintf(file, "Alpha is not found\n");
	fclose(file);
}
