#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* weight = this->blobs_[0]->gpu_data(); //11111
  for (int i = 0; i < 1; ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();  //11111
	const Dtype* bottom_data_mask = bottom[1]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
	Dtype* top_data_mask = top[1]->mutable_cpu_data();
	const int* kernel_shape_data = this->kernel_shape_.cpu_data();
	const int* stride_data = this->stride_.cpu_data();
	const int* pad_data = this->pad_.cpu_data();
	const int h_offset = (kernel_shape_data[0] - 1) / 2 - pad_data[0];
	const int w_offset = (kernel_shape_data[1] - 1) / 2 - pad_data[1];
	for (int h = 0; h < this->output_shape_mask_[0]; h++){
		for (int w = 0; w < this->output_shape_mask_[1]; w++){
			top_data_mask[h*this->output_shape_mask_[1] + w] = bottom_data_mask[h_offset* this->input_shape(2)
				+ h*stride_data[0] * this->input_shape(1) + w_offset + w*stride_data[1]];
		}
	}
	int find_1_num = caffe_cpu_asum(this->output_shape_mask_[0] * this->output_shape_mask_[1], top_data_mask);
	if (find_1_num == 0)
	{
		top_data_mask[0] = 1;
	}
	//LOG(INFO) << "Debuf here:over for\n";
	const Dtype* top_data_masked = top[1]->cpu_data();
	//const Dtype* top_data_masked = top[1]->cpu_data();
    for (int n = 0; n < 1; ++n) {
		this->forward_gpu_gemm_mask(bottom_data + n*this->bottom_dim_, weight,
			top_data + n*this->top_dim_, top_data_masked );

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
