/*
 * output.cpp
 *
 *  Created on: Apr 15, 2011
 *      Author: ludwig
 */

/*
 Copyright 2011 PureSol Technologies

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
#include <stdio.h>
#include "opencl_output.h"

void outputLeft(unsigned int m, unsigned int maxM)
{
	if (m == 0)
	{
		if (maxM == 1)
		{
			printf("(\t");
		}
		else
		{
			printf("/\t");
		}
	}
	else
	{
		if (m == maxM - 1)
		{
			printf("\\\t");
		}
		else
		{
			printf("|\t");
		}
	}
}

void outputRight(unsigned int m, unsigned int maxM)
{
	if (m == 0)
	{
		if (maxM == 1)
		{
			printf(")\t");
		}
		else
		{
			printf("\\\t");
		}
	}
	else
	{
		if (m == maxM - 1)
		{
			printf("/\t");
		}
		else
		{
			printf("|\t");
		}
	}
	printf("\n");
}

void printMatrixS(const MatrixOrder order, const int m, const int n, float *a,
		const int lda)
{
	for (int _m = 0; _m < m; ++_m)
	{
		outputLeft(_m, m);
		for (int _n = 0; _n < n; ++_n)
		{
			if (order == ColumnMajor)
			{
				printf("%e\t", a[_n * lda + _m]);
			}
			else
			{
				printf("%e\t", a[_m * lda + _n]);
			}
		}
		outputRight(_m, m);
	}
}

void printMatrixD(const MatrixOrder order, const int m, const int n, double *a,
		const int lda)
{
	for (int _m = 0; _m < m; ++_m)
	{
		if (_m == 0)
		{
			if (m == 1)
			{
				printf("(\t");
			}
			else
			{
				printf("/\t");
			}
		}
		else
		{
			if (_m == m - 1)
			{
				printf("\\\t");
			}
			else
			{
				printf("|\t");
			}
		}
		for (int _n = 0; _n < n; ++_n)
		{
			if (order == ColumnMajor)
			{
				printf("%e\t", a[_n * lda + _m]);
			}
			else
			{
				printf("%e\t", a[_m * lda + _n]);
			}
		}
		if (_m == 0)
		{
			if (m == 1)
			{
				printf(")\t");
			}
			else
			{
				printf("\\\t");
			}
		}
		else
		{
			if (_m == m - 1)
			{
				printf("/\t");
			}
			else
			{
				printf("|\t");
			}
		}
		printf("\n");
	}
}
