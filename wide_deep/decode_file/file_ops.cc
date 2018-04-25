/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionOrConstant;

REGISTER_OP("DecodeFileServe")
    .Input("records: string")
    .Input("record_defaults: OUT_TYPE")
    .Output("output: OUT_TYPE")
    .Attr("output_size: list(int)")
    .Attr("OUT_TYPE: list({float,double,int32,int64,string})")
    .Attr("field_outer_delim: string = ','")
    .Attr("field_inner_delim: string = ':'")
    .Attr("use_quote_delim: bool = true")
    .Attr("na_value: string = ''")
    .SetShapeFn([](InferenceContext* c) {
      // Validate the record_defaults inputs.
      for (int i = 1; i < c->num_inputs(); ++i) {
        ShapeHandle v;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &v));
        if (c->Value(c->Dim(v, 0)) > 1) {
          return errors::InvalidArgument(
              "Shape of a default must be a length-0 or length-1 vector");
        }
      }

      // Propagate shape of the records input.
      //for (int i = 0; i < c->num_outputs(); ++i) c->set_output(i, c->input(0));
      std::vector<int> output_size;
      c->GetAttr("output_size", &output_size);
      //std::cout<<"output_size" << output_size.size()<<std::endl;
      for (int i = 0; i < c->num_outputs(); ++i) {
        //c->set_output(i, c->input(0));
        //ShapeHandle s = c->MakeShape({DimensionOrConstant(output_size[i]), DimensionOrConstant(output_size[i])});
        
        //ShapeHandle s = c->MakeShape({DimensionOrConstant(output_size[i])});
        //c->set_output(i, s);
        //std::cout<<"-----"<<c->Dim(c->input(0), 0)<<std::endl;
        
        //std::cout<<"i="<<i<<" size="<<output_size[i]<<std::endl;

        ShapeHandle s;
        c->set_output(i, c->Matrix(c->Dim(c->input(0), 0), output_size[i]));
        
        //if(c->WithRank(c->input(0), 1, &s) == Status::OK()) {
        //  std::cout<<"i="<<i<<" size="<<output_size[i]<<std::endl;
        //  c->set_output(i, c->MakeShape({output_size[i]}));
        //}
        //else {
        //  c->set_output(i, c->Matrix(c->Dim(c->input(0), 0), output_size[i]));
        //}
      }

      return Status::OK();
    })
    .Doc(R"doc(
Convert file records to tensors.
)doc");

}  // namespace tensorflow
