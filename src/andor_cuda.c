#include <THC/THC.h>

#include "andor_cuda_kernel.h"

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

/**
int andor_forward_cuda(
                THCudaTensor *positive_inputs_batch,

                THCudaTensor *positive_and_negative_errors_batch,

                THCudaTensor *positive_and_negative_weights,

                THCudaTensor *positive_and_negative_errors_output_batch,

		        THCudaTensor *positive_output_batch,
		        )
*/

int andor_forward_cuda(THCudaTensor *a_tensor, THCudaTensor *b_tensor,
		       THCudaTensor *output, int x, int y)
{
//  if (!THCudaTensor_isSameSizeAs(state, input1, input2))
//    return 0;
//  THCudaTensor_resizeAs(state, output, input1);
//  THCudaTensor_cadd(state, output, input1, 1.0, input2);
//
  float *a = THCudaTensor_data(state, a_tensor);
  float *b = THCudaTensor_data(state, b_tensor);
  float *c = THCudaTensor_data(state, output);
  cudaStream_t stream = THCState_getCurrentStream(state);

  andor_cuda(c, a, b, x, y, stream);

  return 1;
}

int andor_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input)
{
  THCudaTensor_resizeAs(state, grad_input, grad_output);
  THCudaTensor_fill(state, grad_input, 1);
  return 1;
}
