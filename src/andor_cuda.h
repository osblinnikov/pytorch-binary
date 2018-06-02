int andor_forward_cuda(THCudaTensor *input1, THCudaTensor *input2,
		       THCudaTensor *output);
int andor_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input);
