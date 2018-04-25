/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

// See docs in ../ops/parsing_ops.cc.
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

//template <typename T>
class DecodeFileServeOp : public OpKernel {
 public:
  explicit DecodeFileServeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string inner_delim;
    string outer_delim;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("OUT_TYPE", &out_type_));
    OP_REQUIRES(ctx, out_type_.size() < std::numeric_limits<int>::max(),
                errors::InvalidArgument("Out type too large"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_size", &output_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_inner_delim", &inner_delim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_outer_delim", &outer_delim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_quote_delim", &use_quote_delim_));
    OP_REQUIRES(ctx, inner_delim.size() == 1,
                errors::InvalidArgument("field_inner_delim should be only 1 char"));
    OP_REQUIRES(ctx, outer_delim.size() == 1,
                errors::InvalidArgument("field_outer_delim should be only 1 char"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("na_value", &na_value_));

    inner_delim_ = inner_delim[0];
    outer_delim_ = outer_delim[0];
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* records;
    OpInputList record_defaults;
    OpInputList record_skip;

    OP_REQUIRES_OK(ctx, ctx->input("records", &records));
    OP_REQUIRES_OK(ctx, ctx->input_list("record_defaults", &record_defaults));
    //OP_REQUIRES_OK(ctx, ctx->input_list("record_skip", &record_skip));

    for (int i = 0; i < record_defaults.size(); ++i) {
      OP_REQUIRES(ctx, record_defaults[i].NumElements() < 2,
                  errors::InvalidArgument(
                      "There should only be 1 default per field but field ", i,
                      " has ", record_defaults[i].NumElements()));
    }

    auto records_t = records->flat<string>();
    int64 records_size = records_t.size();
    // record_size is example size
    //OP_REQUIRES(ctx, records_size == 1, errors::InvalidArgument("input records size should only be 1"));

    //std::cout<<"records size"<< static_cast<int>(records_size) << std::endl;
    //std::cout<<"records shape"<<records->shape().dims()<<std::endl;
    //std::cout<<"records shape:"<<records->shape().dim_size(0)<<std::endl;
    //std::cout<<"records shape:"<<records->shape().dim_size(1)<<std::endl;

    OpOutputList output;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &output));

    for (int i = 0; i < static_cast<int>(out_type_.size()); ++i) {
      Tensor* out = nullptr;
      //OP_REQUIRES_OK(ctx, output.allocate(i, records->shape(), &out));

      int32 field_size = output_size_[i];
      //OP_REQUIRES_OK(ctx, output.allocate(i, TensorShape({field_size}), &out));
      OP_REQUIRES_OK(ctx, output.allocate(i, TensorShape({records_size, field_size}), &out));

      //if(static_cast<int>(records_size) == 1)
      //  OP_REQUIRES_OK(ctx, output.allocate(i, TensorShape({field_size}), &out));
      //else
      //  OP_REQUIRES_OK(ctx, output.allocate(i, TensorShape({records_size, field_size}), &out));
    }

    for (int64 i = 0; i < records_size; ++i) {
      const StringPiece record(records_t(i));
      
      std::vector<std::vector<string> > fields;
      ExtractFields(ctx, record, outer_delim_, inner_delim_, &fields);
      OP_REQUIRES(ctx, fields.size() == out_type_.size(),
                  errors::InvalidArgument("Expect ", out_type_.size(),
                                          " fields but have ", fields.size(),
                                          " in record ", i));
      // Check each field in the record, support DT_INT32
      for (int f = 0; f < static_cast<int>(out_type_.size()); ++f) {
        const DataType& dtype = out_type_[f];
        //int field_size = static_cast<int>(output_size[f].flat<int32>()(0));
        int field_size = output_size_[f];
        //std::cout << "field:" << f << " field_size = " << static_cast<int>(field_size)<<std::endl;
        OP_REQUIRES(ctx, fields[f].size() <= output_size_[f],
                  errors::InvalidArgument("Input field: ", f, " size = ", fields[f].size(),
                                          " exceed field output size: ", output_size_[f],
                                          " in record ", i));

        switch (dtype) {
          case DT_INT32: {
            auto field_output = output[f]->flat<int32>();

            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f][0] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));

              field_output(i) = record_defaults[f].flat<int32>()(0);
            } else {
              for(int j=0; j<fields[f].size(); j++) {
                int32 value;
                OP_REQUIRES(ctx, strings::safe_strto32(fields[f][j], &value),
                            errors::InvalidArgument("Field ", f, " in record ", i,
                                                    " is not a valid int32: ",
                                                    fields[f][0]));
                field_output(i*field_size+j) = value;
              }
            }
            break;
          }
          case DT_INT64: {
            auto field_output = output[f]->flat<int64>();

            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f][0] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));
              field_output(i) = record_defaults[f].flat<int32>()(0);
            } else {
              for(int j=0; j<fields[f].size(); j++) {
                int64 value;
                OP_REQUIRES(ctx, strings::safe_strto64(fields[f][j], &value),
                            errors::InvalidArgument("Field ", f, " in record ", i,
                                                    " is not a valid int64: ",
                                                    fields[f][0]));
                field_output(i*field_size+j) = value;
              }
            }
            break;
          }
          case DT_FLOAT: {
            auto field_output = output[f]->flat<float>();

            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f][0] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));
              field_output(i) = record_defaults[f].flat<float>()(0);
            } else {
              for(int j=0; j<fields[f].size(); j++) {
                float value;
                OP_REQUIRES(ctx, strings::safe_strtof(fields[f][0].c_str(), &value),
                            errors::InvalidArgument("Field ", f, " in record ", i,
                                                    " is not a valid float: ",
                                                    fields[f][0]));
                field_output(i*field_size+j) = value;
              }
            }
            break;
          }
          case DT_DOUBLE: {
            auto field_output = output[f]->flat<double>();

            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f][0] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));
              field_output(i) = record_defaults[f].flat<double>()(0);
            } else {
              for(int j=0; j<fields[f].size(); j++) {
                double value;
                OP_REQUIRES(ctx, strings::safe_strtod(fields[f][0].c_str(), &value),
                            errors::InvalidArgument("Field ", f, " in record ", i,
                                                    " is not a valid double: ",
                                                    fields[f][0]));
                field_output(i*field_size+j) = value;
              }
            }
            break;
          }
          case DT_STRING: {
            //auto field_output = output[f]->matrix<string>();
            auto field_output = output[f]->flat<string>();

            // If this field is empty or NA value, check if default is given:
            // If yes, use default value; Otherwise report error.
            if (fields[f].empty() || fields[f][0] == na_value_) {
              OP_REQUIRES(ctx, record_defaults[f].NumElements() == 1,
                          errors::InvalidArgument(
                              "Field ", f,
                              " is required but missing in record ", i, "!"));
              field_output(i) = record_defaults[f].flat<string>()(0);
            } else {
              for(int j=0; j<fields[f].size(); j++) {
                field_output(i*field_size+j) = fields[f][j];
              }
            }
            break;
          }
          default:
            OP_REQUIRES(ctx, false,
                        errors::InvalidArgument("csv: data type ", dtype,
                                                " not supported in field ", f));
        }
      }
    }
  }

 private:
  std::vector<DataType> out_type_;
  std::vector<int> output_size_;
  char inner_delim_;
  char outer_delim_;
  bool use_quote_delim_;
  string na_value_;

  void ExtractFields(OpKernelContext* ctx, StringPiece input, char outer_delim_, char inner_delim_,
                     std::vector<std::vector<string> >* result) {
    //std::cout<<"input:"<<input << " size=" << input.size() <<std::endl;
    int64 current_idx = 0;
    if (!input.empty()) {
      while (static_cast<size_t>(current_idx) < input.size()) {
        if (input[current_idx] == '\n' || input[current_idx] == '\r') {
          current_idx++;
          continue;
        }

        bool quoted = false;
        if (use_quote_delim_ && input[current_idx] == '"') {
          quoted = true;
          current_idx++;
        }

        // This is the body of the field;
        string value;
        std::vector<string> field;
        if (!quoted) {
          while (static_cast<size_t>(current_idx) < input.size()) {
            if(input[current_idx] == outer_delim_) {
              //std::cout<<"outer value="<<value<<std::endl;
              field.push_back(std::move(value));
              result->push_back(std::move(field));
            } else if(input[current_idx] == inner_delim_) {
              //std::cout<<"inner value="<<value<<std::endl;
              field.push_back(std::move(value));
            } else {
              value += input[current_idx];
            }
            current_idx++;
          }

          // Go to next field or the end
          //current_idx++;
        } else if (use_quote_delim_) {
          // Quoted field needs to be ended with '"' and delim or end
          //while (
          //    (static_cast<size_t>(current_idx) < input.size() - 1) &&
          //    (input[current_idx] != '"' || input[current_idx + 1] != delim_)) {
          //  if (input[current_idx] != '"') {
          //    value += input[current_idx];
          //    current_idx++;
          //  } else {
          //    OP_REQUIRES(
          //        ctx, input[current_idx + 1] == '"',
          //        errors::InvalidArgument("Quote inside a string has to be "
          //                                "escaped by another quote"));
          //    value += '"';
          //    current_idx += 2;
          //  }
          //}

          //OP_REQUIRES(
          //    ctx, (static_cast<size_t>(current_idx) < input.size() &&
          //          input[current_idx] == '"' &&
          //          (static_cast<size_t>(current_idx) == input.size() - 1 ||
          //           input[current_idx + 1] == delim_)),
          //    errors::InvalidArgument("Quoted field has to end with quote "
          //                            "followed by delim or end"));

          //current_idx += 2;
        }

        field.push_back(std::move(value));
        result->push_back(std::move(field));
        //std::cout << "restult_size:" << result->size()<<std::endl;
        //std::cout << "field_size:" << field.size()<<std::endl;
        //result->push_back(field);
      }

      // Check if the last field is missing
      //if (input[input.size() - 1] == delim_) result->push_back(string());
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("DecodeFileServe").Device(DEVICE_CPU), DecodeFileServeOp);

}  // namespace tensorflow
