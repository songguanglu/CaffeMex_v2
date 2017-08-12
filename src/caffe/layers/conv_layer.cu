#include <vector>
#include <algorithm>
#include "caffe/layers/conv_layer.hpp"

namespace caffe {


	template <typename Dtype>
	__global__ void MaskCopy(const int n, const Dtype* in, const int* src, Dtype* out, const int* dst) {
		CUDA_KERNEL_LOOP(index, n) {
			out[dst[index]] = in[src[index]];
		}
	}

	template <typename Dtype>
	void ConvolutionLayer<Dtype>::forward_gpu_gemm_mask(const Dtype* input,
		const Dtype* weights, Dtype* output, const Dtype* mask_input, bool skip_im2col) {
		//clock_t start, end, all_start, all_end;
		const Dtype* col_buff = input;
//		double dur;
		//start = clock();
		//all_start = clock();
		const int height = conv_input_shape_.cpu_data()[1];
		const int width = conv_input_shape_.cpu_data()[2];
		const int kernel_h = kernel_shape_.cpu_data()[0];
		const int kernel_w = kernel_shape_.cpu_data()[1];
		const int pad_h = pad_.cpu_data()[0];
		const int pad_w = pad_.cpu_data()[1];
		const int stride_h = stride_.cpu_data()[0];
		const int stride_w = stride_.cpu_data()[1];
		const int dilation_h = dilation_.cpu_data()[0];
		const int dilation_w = dilation_.cpu_data()[1];
		int height_col = (height + 2 * pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		int width_col = (width + 2 * pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		int validnum = caffe_cpu_asum(height_col*width_col, mask_input);
		/*int* src_index = new int[validnum*kernel_dim_];
		
		int* dst_index = new int[validnum*kernel_dim_];
		int* src_fin_index = new int[validnum*conv_out_channels_];
		int* dst_fin_index = new int[validnum*conv_out_channels_];*/
		src_index_.Reshape(validnum*kernel_dim_, 1, 1, 1);
		dst_index_.Reshape(validnum*kernel_dim_,1,1,1);
		src_fin_index_.Reshape(validnum*conv_out_channels_, 1, 1, 1);
		dst_fin_index_.Reshape(validnum*conv_out_channels_, 1, 1, 1);
		int* src_index = src_index_.mutable_cpu_data();
		int* dst_index = dst_index_.mutable_cpu_data();
		int* src_fin_index = src_fin_index_.mutable_cpu_data();
		int* dst_fin_index = dst_fin_index_.mutable_cpu_data();
		int cnt = 0;
		pass_idx_.Reshape(validnum, 1, 1, 1);
		buffer_col_.Reshape(kernel_dim_, validnum, 1, 1);
		Dtype* buffer_col_data = buffer_col_.mutable_gpu_data();
		//Dtype* buffer_col_data = buffer_col_.mutable_cpu_data();
		output_buffer_.Reshape(conv_out_channels_, validnum, 1, 1);
	//	end = clock();
//		dur = (double)(end - start);
		//LOG(INFO) << "the base_conv_layer before im2col using time:" << dur / CLOCKS_PER_SEC;
		//start = clock();
		int idx = 0;
		if (!is_1x1_) {
			if (!skip_im2col) {
				conv_im2col_gpu(input, col_buffer_.mutable_gpu_data()); //here 11111
			}
			col_buff = col_buffer_.gpu_data();
		}
			//	LOG(INFO) << "Debuf here:finish im2col\n";
		//	end = clock();

			
		//	dur = (double)(end - start);
		//	LOG(INFO) << "the base_conv_layer im2col using time:" << dur / CLOCKS_PER_SEC;
		//	google::FlushLogFiles(google::INFO);
			//	col_buff = col_buffer_.cpu_data();
			//generate new trans respond to mask 1
		//	google::FlushLogFiles(google::INFO);
		//	start = clock();
			for (int h = 0; h < height_col; h++){
				for (int w = 0; w < width_col; w++){
					if (mask_input[h*width_col + w] >= 1)
					{
						for (int temp = 0; temp < kernel_dim_; temp++){
							src_index[cnt] = temp*height_col*width_col + h*width_col + w;
							dst_index[cnt] = temp*validnum + idx;
							cnt++;
						}
						idx += 1;
					}
				}
			}
			const int* src_use_index = src_index_.cpu_data();
			const int* dst_use_index = dst_index_.cpu_data();
			MaskCopy<Dtype> << <CAFFE_GET_BLOCKS(validnum*kernel_dim_), CAFFE_CUDA_NUM_THREADS >> >(
				validnum*kernel_dim_, col_buff, src_use_index, buffer_col_data, dst_use_index);
			CUDA_POST_KERNEL_CHECK;

		//	end = clock();
		//	dur = (double)(end - start);
		//	LOG(INFO) << "the base_conv_layer 418-429 using time:" << dur / CLOCKS_PER_SEC;
		//	google::FlushLogFiles(google::INFO);

		//Dtype* output_buffer_data = output_buffer_.mutable_gpu_data();
		//const Dtype* buffer_col_data_com = buffer_col_.gpu_data();
		//start = clock();
		Dtype* output_buffer_data = output_buffer_.mutable_gpu_data();
		const Dtype* buffer_col_data_com = buffer_col_.gpu_data();

		for (int g = 0; g < group_; ++g) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
				group_, validnum, kernel_dim_,
				(Dtype)1., weights + weight_offset_ * g, buffer_col_data_com + col_offset_ * g,
				(Dtype)0., output_buffer_data + conv_out_channels_* validnum / group_* g); //here11111
		}
		//end = clock();
		//dur = (double)(end - start);
		//LOG(INFO) << "the base_conv_layer 435-440 using time:" << dur / CLOCKS_PER_SEC;
		//google::FlushLogFiles(google::INFO);
		//generate new output for mask 0
		//start = clock();
		caffe_gpu_set(output_offset_, Dtype(0), output);
		idx = 0;
		//LOG(INFO) << "Finish copy data";
		//google::FlushLogFiles(google::INFO);
		const Dtype* output_buffer_data_fin = output_buffer_.gpu_data();
		cnt = 0;
		//google::FlushLogFiles(google::INFO);
		for (int h = 0; h < height_col; h++){
			for (int w = 0; w < width_col; w++){
				if (mask_input[h*width_col + w] >= 1)
				{
					for (int temp = 0; temp < conv_out_channels_; temp++){
						src_fin_index[cnt] = temp*validnum + idx;
						dst_fin_index[cnt] = temp*height_col*width_col + h*width_col + w;
						cnt++;
					}
					idx += 1;
				}
			}
		}
		const int* src_fin_use_index = src_fin_index_.cpu_data();
		const int* dst_fin_use_index = dst_fin_index_.cpu_data();
		MaskCopy<Dtype> << <CAFFE_GET_BLOCKS(validnum*conv_out_channels_), CAFFE_CUDA_NUM_THREADS >> >(
			validnum*conv_out_channels_, output_buffer_data_fin, src_fin_use_index, output, dst_fin_use_index);
		CUDA_POST_KERNEL_CHECK;
		//end = clock();
		//dur = (double)(end - start);
		//LOG(INFO) << "the base_conv_layer 446-457 using time:" << dur / CLOCKS_PER_SEC;
		//google::FlushLogFiles(google::INFO);
		//all_end = clock();
		//dur = (double)(all_end - all_start);
		//LOG(INFO) << "the gemm_mask inner using time:" << dur / CLOCKS_PER_SEC;
	}


template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	//clock_t start, end, mask_start,mask_end,start_line,end_line;
	//double dur;
	//start = clock();
	LARGE_INTEGER t1, t2, tc,x1,x2;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);
	const Dtype* weight = this->blobs_[0]->gpu_data(); //11111
  for (int i = 0; i < 1; ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();  //11111
	const Dtype* bottom_data_mask = bottom[1]->cpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
	Dtype* top_data_mask = top[1]->mutable_cpu_data();
	const int* kernel_shape_data = this->kernel_shape_.cpu_data();
	const int* stride_data = this->stride_.cpu_data();
	const int* pad_data = this->pad_.cpu_data();
	const int h_offset = (kernel_shape_data[0] - 1) / 2 - pad_data[0];
	const int w_offset = (kernel_shape_data[1] - 1) / 2 - pad_data[1];
	//start_line = clock();
	for (int h = 0; h < this->output_shape_mask_[0]; h++){
		for (int w = 0; w < this->output_shape_mask_[1]; w++){
			top_data_mask[h*this->output_shape_mask_[1] + w] = bottom_data_mask[h_offset* this->input_shape(2)
				+ h*stride_data[0] * this->input_shape(1) + w_offset + w*stride_data[1]];
		}
	}
	//end_line = clock();
	//dur =(double)(end_line - start_line);
	//LOG(INFO) << "the conv_layer.cu 21-26 using time:" << dur / CLOCKS_PER_SEC;
	int find_1_num = caffe_cpu_asum(this->output_shape_mask_[0] * this->output_shape_mask_[1], top_data_mask);
	if (find_1_num == 0)
	{
		top_data_mask[0] = 1;
	}
	//LOG(INFO) << "Debuf here:over for\n";
	const Dtype* top_data_masked = top[1]->cpu_data();
	//const Dtype* top_data_masked = top[1]->cpu_data();
    for (int n = 0; n < 1; ++n) {
	//	mask_start = clock();
		QueryPerformanceCounter(&x1);
		this->forward_gpu_gemm_mask(bottom_data + n*this->bottom_dim_, weight,
			top_data + n*this->top_dim_, top_data_masked );
		QueryPerformanceCounter(&x2);
		LOG(INFO) << "forward_gpu_gemm_mask_using_time:" <<( (x2.QuadPart - x1.QuadPart)*1.0 / tc.QuadPart);
	//	mask_end = clock();
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
 // end = clock();
//  dur =(double)(mask_end - mask_start);
 // LOG(INFO) << "the mask conv layer forward_gpu_gemm_mask using time:" << dur / CLOCKS_PER_SEC;
  //dur =(double)(end - start);
 // LOG(INFO) << "the mask conv layer forward_gpu using time:" << dur / CLOCKS_PER_SEC;
  QueryPerformanceCounter(&t2);
  LOG(INFO) << "forward_gpu_using_time:" << ((t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);
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
