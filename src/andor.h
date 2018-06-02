int andor_forward(THFloatTensor *input1, THFloatTensor *input2,
		       THFloatTensor *output);
int andor_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);
