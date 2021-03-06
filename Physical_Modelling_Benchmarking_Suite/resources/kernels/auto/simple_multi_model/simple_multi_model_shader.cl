int rem(int x, int y)
{
    return (x % y + y) % y;
}

__kernel
void fdtdKernel(__global int* idGrid, __global float* modelGrid, __global float* boundaryGrid, int idxRotate, int idxSample, __global float* input, __global float* output, int inputPosition, int outputPosition, float stringMu, float stringLambda)
{
	//Rotation Index into model grid//
	int gridSize = get_global_size(0) * get_global_size(1);
    
	int rotation0 = gridSize * rem(idxRotate+0, 3);
	int rotationM1 = gridSize * rem(idxRotate+-1, 3);
	int rotation1 = gridSize * rem(idxRotate+1, 3);
	
    
	//Get index for current and neighbouring nodes//
	int t0x0y0Idx = rotation0 + ((get_global_id(1)+0) * get_global_size(0) + get_global_id(0)+0);
	int t0x1y0Idx = rotation0 + ((get_global_id(1)+1) * get_global_size(0) + get_global_id(0)+0);
	int t0xM1y0Idx = rotation0 + ((get_global_id(1)+-1) * get_global_size(0) + get_global_id(0)+0);
	int tM1x0y0Idx = rotationM1 + ((get_global_id(1)+0) * get_global_size(0) + get_global_id(0)+0);
	
	int t1x0y0Idx = rotation1 + ((get_global_id(1)) * get_global_size(0) + get_global_id(0));

	//Boundary condition evaluates neighbours in preperation for equation//
	//@ToDo - Make new timestep value autogenerated?//
	float t1x0y0;
	float t0x0y0;
	float t0x1y0;
	float t0xM1y0;
	float tM1x0y0;
	
		t0x0y0 = modelGrid[t0x0y0Idx];
	tM1x0y0 = modelGrid[tM1x0y0Idx];
	

	int centreIdx = (get_global_id(1)) * get_global_size(0) + get_global_id(0);
	if(boundaryGrid[centreIdx] > 0.01)
	{
		t0xM1y0 = 0;
		t0x1y0 = 0;
		
	}
	else
	{
		t0xM1y0 = modelGrid[t0xM1y0Idx];
		t0x1y0 = modelGrid[t0x1y0Idx];
		
	}
	
	//Calculate the next pressure value//
	if(idGrid[centreIdx] == 0)
{t1x0y0 = 0.0;}
if(idGrid[centreIdx] == 1) {
		t1x0y0 = (((2*t0x0y0)+((stringLambda*stringLambda)*(t0x1y0-(2*t0x0y0)+t0xM1y0))-tM1x0y0)*(1.0/(stringMu+1.0)));
}
	if(idGrid[centreIdx] == 2) {
		t1x0y0 = (((2*t0x0y0)+((stringLambda*stringLambda)*(t0x1y0-(2*t0x0y0)+t0xM1y0))-tM1x0y0)*(1.0/(stringMu+1.0)));
}
	if(idGrid[centreIdx] == 3) {
		t1x0y0 = (((2*t0x0y0)+((stringLambda*stringLambda)*(t0x1y0-(2*t0x0y0)+t0xM1y0))-tM1x0y0)*(1.0/(stringMu+1.0)));
}
	if(idGrid[centreIdx] == 4) {
		t1x0y0 = (((2*t0x0y0)+((stringLambda*stringLambda)*(t0x1y0-(2*t0x0y0)+t0xM1y0))-tM1x0y0)*(1.0/(stringMu+1.0)));
}
	if(idGrid[centreIdx] == 5) {
		t1x0y0 = (((2*t0x0y0)+((stringLambda*stringLambda)*(t0x1y0-(2*t0x0y0)+t0xM1y0))-tM1x0y0)*(1.0/(stringMu+1.0)));
}
	if(idGrid[centreIdx] == 6) {
		t1x0y0 = (((2*t0x0y0)+((stringLambda*stringLambda)*(t0x1y0-(2*t0x0y0)+t0xM1y0))-tM1x0y0)*(1.0/(stringMu+1.0)));
}
	if(idGrid[centreIdx] == 7) {
		t1x0y0 = (((2*t0x0y0)+((stringLambda*stringLambda)*(t0x1y0-(2*t0x0y0)+t0xM1y0))-tM1x0y0)*(1.0/(stringMu+1.0)));
}
	if(idGrid[centreIdx] == 8) {
		t1x0y0 = (((2*t0x0y0)+((stringLambda*stringLambda)*(t0x1y0-(2*t0x0y0)+t0xM1y0))-tM1x0y0)*(1.0/(stringMu+1.0)));
}
	if(idGrid[centreIdx] == 9) {
		t1x0y0 = (((2*t0x0y0)+((stringLambda*stringLambda)*(t0x1y0-(2*t0x0y0)+t0xM1y0))-tM1x0y0)*(1.0/(stringMu+1.0)));
}
	if(idGrid[centreIdx] == 10) {
		t1x0y0 = (((2*t0x0y0)+((stringLambda*stringLambda)*(t0x1y0-(2*t0x0y0)+t0xM1y0))-tM1x0y0)*(1.0/(stringMu+1.0)));
}
	;
	
	//If the cell is the listener position, sets the next sound sample in buffer to value contained here//
	if(centreIdx == outputPosition)
	{
		output[idxSample]= t0x0y0;    //@ToDo - Make current timestep centre point auto generated?
	}
	
	if(centreIdx == inputPosition)	//If the position is an excitation...
	{
		t1x0y0 += input[idxSample];	//Input excitation value into point. Then increment to next excitation in next iteration.
	}
	
	modelGrid[t1x0y0Idx] = t1x0y0;
}