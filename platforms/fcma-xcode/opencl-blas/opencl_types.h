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

#pragma once

typedef enum
{
	opencl_success,
	opencl_not_implemented,
	opencl_failure,
	opencl_architecture_mismatch
} OpenCLStatus;

typedef enum
{
	RowMajor = 0, ColumnMajor = 1
} MatrixOrder;

typedef enum
{
	NoTranspose = 0, Transpose = 1
} TransposeMatrix;

