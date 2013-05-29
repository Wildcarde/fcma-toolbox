threadRows = 16
threadCols = 16
repRows = 6
repCols = 6

if threadRows != threadCols:
	print '#error "sgemm kernel build with threadRows != threadCols"'

print '#pragma once'
print '#ifdef __DEVICE__'
print '#define LDAS ' + str(threadRows * repRows + 1)
print '#define LDBS ' + str(threadCols + 1)
print ''
print '__attribute__((reqd_work_group_size(' + str(threadRows) + ', ' + str(threadCols) + ', 1)))'
print '__kernel void sgemmNN(int n, int m, int k, float alpha, __global float* a, int lda, __global float* b, int ldb, float beta, __global float* c, int ldc)'
print '{'

for i in range(0, repRows):
	for j in range(0, repCols):
		print '  float c' + str(i * threadRows) + '_' + str(j * threadRows) + ' = 0.0f;'

print ''

for i in range(0, repRows):
	print '  float a' + str(i * threadRows) + ';'

print ''

for i in range(0, repCols):
	print '  float b' + str(i * threadCols) + ';'

print ''
print '  __local float aShared[LDAS * ' + str(threadCols) + '];'
print '  __local float bShared[LDBS * ' + str(threadRows * repRows) + '];'
print ''
print '  const int rowBlock = get_local_id(0);'
print '  const int colBlock = get_local_id(1);'
print ''
print '  const int rowC = get_group_id(0) * ' + str(threadRows * repRows) + ';'
print '  const int colC = get_group_id(1) * ' + str(threadCols * repCols) + ';'
print ''
print '  __global float* blockC;'
print ''
print '  for(int i = 0; i < k; i+= ' + str(threadCols) + ')'
print '  {'
print '    #pragma unroll ' + str(repRows)
print '    for(int l = 0; l < ' + str(threadRows * repRows) + '; l+= ' + str(threadRows) + ')'
print '    {'
print '      aShared[rowBlock + l + LDAS * colBlock] = a[rowC + rowBlock + l + (colBlock + i) * lda];'

if repCols != repCols:
	print '    }'
	print ''
	print '    #pragma unroll ' + str(repCols)
	print '    for(int l = 0; l < ' + str(threadCols * repCols) + '; l+= ' + str(threadCols) + ')'
	print '    {'
print '      bShared[colBlock + LDBS * (rowBlock + l)] = b[colBlock + i + (colC + rowBlock + l) * ldb];'
print '    }'
print ''
print '    barrier(CLK_LOCAL_MEM_FENCE);'
print ''
print '    for(int l = 0; l <' + str(threadCols)  + '; l++)'
print '    {'

for i in range (0, repRows):
	print '      a' + str(i * threadRows) + ' = aShared[rowBlock + ' + str(i * threadRows) + ' + LDAS * l];'

print ''

for i in range (0, repCols):
	print '      b' + str(i * threadCols) + ' = bShared[l + LDBS * (colBlock + ' + str(i * threadCols) + ')];'

print ''

for i in range(0, repRows):
	for j in range(0, repCols):
		print '      c' + str(i * threadRows) + '_' + str(j * threadCols) + ' = fma(a' + str(i * threadRows) + ', b' + str(j * threadCols) + ', c' + str(i * threadRows) + '_' + str(j * threadCols) + ');'

print '    }'
print ''
print '    barrier(CLK_LOCAL_MEM_FENCE);'

print '  }'
print ''
print '  blockC = &c[rowC + ldc * (colC + colBlock)];'
print ''

print '  if(rowC + ' + str(repRows * threadRows - 1) + ' < m && colC + ' + str(repCols * threadCols - 1) + ' < n)'
print '  {'
for j in range(0, repCols):
	for i in range(0, repRows):
		print '    blockC[rowBlock + ' + str(i * threadRows) + '] = alpha * c' + str(i * threadRows) + '_' + str(j * threadCols) + ' + beta * blockC[rowBlock + ' + str(i * threadRows) + '];'
	print ''
	print '    blockC += ' + str(threadCols) + ' * ldc;'
	print ''
print '  }'
print ''
print '}'
print ''
print '#else'
print '#define SGEMMTHREADROWS ' + str(threadRows)
print '#define SGEMMTHREADCOLS ' + str(threadCols)
print '#define SGEMMROWREP ' + str(repRows)
print '#define SGEMMCOLREP ' + str(repCols)
print '#endif'
