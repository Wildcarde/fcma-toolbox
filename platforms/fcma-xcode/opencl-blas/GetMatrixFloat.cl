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

float getMatrixFloat(FloatMatrix mat, unsigned int row, unsigned int col)
{
	if ((row >= mat.m) || (col >= mat.n))
	{
		return 0.0;
	}
	if (mat.colMajor == 0)
	{
		/* row major data access */
		if (mat.transposed == 0)
		{
			/* non-transposed row major data access */
			return mat.data[row * mat.stride + col];
		}
		else
		{
			/* transposed column major data access */
			return mat.data[col * mat.stride + row];
		}
	}
	else
	{
		/* column major data access */
		if (mat.transposed == 0)
		{
			/* non-transposed column major data access */
			return mat.data[col * mat.stride + row];
		}
		else
		{
			/* transposed column major data access */
			return mat.data[row * mat.stride + col];
		}
	}
}
