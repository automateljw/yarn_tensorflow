from tensorflow.contrib import predictor
import tensorflow as tf
import timeit
import numpy as np
import math

#module_decode_file = tf.load_op_library('./decode_file_op.so')
module_decode_file = tf.load_op_library('src_cpu/decode_file_serve_op.so')

offset_dict = {}
with open('../upload/output/psid_bias.txt') as fd:
  for line in fd:
    ps, value = line.strip().split('\t')
    offset_dict[ps] = float(value)

_COLUMN_NAMES = [
    'label',
    'ps', 'to', 'ar', 'tm', 'wk',
    'os', 'br', 'dv', 'chan', 'chan2',
    'ag', 'gd', 'cmi', 'uch1', 'uch2',
    'uhy', 'uhy1', 'uhy2', 'uhy3', 'rthy', 'rthy1',
    'rthy2', 'rthy3', 'st', 'rtst', 'lastcate',
    'lipv', 'liclk', 'rtpv', 'rtclk', 'net_wifi',
    'hy1', 'hy2', 'hy3', 'hy4', 'cust_uid',
    'deliver_id', 'simid', 'ti', 'fng', 'agt',
    'ip3', 'rthyfreq', 'hyfreq'
]

_COLUMN_SIZESt = [
    1, 
    1, 1, 100, 100, 100,
    100, 100, 100, 100, 100,
    100, 100, 100, 100, 100,
    100, 100, 100, 100, 100,
    100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100,
    100, 100, 100, 100, 100,
    100, 100, 100, 100, 100,
    100, 100, 100
]

fs = dict(zip(_COLUMN_NAMES, _COLUMN_SIZESt))

# load savedmodel
#export_dir = open('export_raw/checkpoint', 'r').read().strip()
#print(export_dir)
export_dir = 'output'

#signature_def = tf.saved_model.signature_def_utils.build_signature_def(
#          inputs={,
#          outputs=outputs,
#          method_name='tensorflow/serving/regress')

predict_fn = predictor.from_saved_model(export_dir, 'predict')

#predict_fn = predictor.from_saved_model(export_dir)
#predictions = predict_fn(input_data)
#predictions = predict_fn({"input":input})

def string_decode_input_predict(predict_fn):
    input=[["0,PDPS000000061522,777000.777020,777,12,4,ios,other,iphone7plus,10,0,603,501,20002%20957%20959,news,news_sh%tech_mobile%auto_null%sports_nba%games_pi%tech_digi%health_disease%ent_music%video_mobile%ent_zy,244x273x523x,244,244x273,244x273x523,null,null,null,null,j1I149%AVwtXc%JNgAnB%eZ89px%80TxnU%lEWGmA%gIGiRr%QShpyD%KLu4HW%7wHTmX,null,null,48,0,48,0,1102:0.0,244,244x286,244x286x296,244x286x296x,6219522904_PINPAI-CPC,3180526,38jIEGglZbIBJbwJUIQzCT,7sDoPyKIUHlBJEdBanDQrM,1601,Ur4,120.18.90,512_5_0%544_5_0%546_5_0%231_5_0%517_5_0%516_5_0%555_5_0%523_5_0,467_0_0%527_5_0%591_0_0%1x2x481_1_0%383_0_0%354_0_0%1x110x503_3_0%244x273x523_5_3%244x397x541_5_0%244x245x516_5_1%555_5_0%512_5_0%546_5_0%1x110x498_0_0%244x286x530_0_0%1x110x500_2_0%244x273x522_0_0%244x286x337x529_4_0%231_5_0%244x245x517_5_0%561_5_0%244x397x544_3_0"]] * 1


    tic = timeit.default_timer()
    predictions = predict_fn({"input":input})
    toc = timeit.default_timer()
    print(predictions)
    print('predict time:', toc - tic)

def string_placeholder_input_predict(predict_fn):
    label=[]
    ps = []
    def handle_feature():
        fd = tf.gfile.Open('test_all.txt')
        input_data = {}
        input_datas = {}
        count = 0
        for line in fd:
            count += 1
            line = line.strip().split(',')
            label.append(int(line[0]))
            ps.append(line[1])
            input_data = dict(zip(_COLUMN_NAMES[1:], line[1:]))
            for k,v in input_data.items():
                if fs[k] == 1:
                    input_data[k] = [input_data[k]]
                else:
                    input_data[k] = input_data[k].split('%') + ['']*(fs[k] - len(input_data[k].split('%')))
            for k,v in input_data.items():
                if k not in input_datas: input_datas[k] = []
                input_datas[k].append(input_data[k])
        fd.close()
        skip_fields = set(['lastcate', 'cust_uid', 'net_wifi', 'ip3', 'deliver_id'])
        for x in skip_fields:
            del input_datas[x]
        print("count:", count)
        return input_datas

    input_data = handle_feature()
    tic = timeit.default_timer()
    predictions = predict_fn(input_data)
    print("label:", np.sum(label))
    print("pred:", predictions['logistic'])
    pred = []
    true_label=[]
    for i, x in enumerate(ps):
       if x not in offset_dict: continue
       pred.append(1/(1+math.exp(-(predictions['logits'][i] + offset_dict[x])))) 
       true_label.append(label[i])
       print("logits:", predictions['logits'][i], "bias", offset_dict[x])
    print("valid label num:", len(true_label))
    print("valid label:", np.sum(true_label))
    print("pred:", np.sum(pred))
    print("pred list:", pred)
    toc = timeit.default_timer()
    print('predict time:', toc - tic)

def string_placeholder_input_predict_test(predict_fn):
    def handle_feature():
        #input_datas = dict(zip(_COLUMN_NAMES[1:], []))
        input_datas = {}
        size_info = dict(zip(_COLUMN_NAMES[1:], _COLUMN_SIZESt[1:]))
        fd = tf.gfile.Open('test.txt')
        for line in fd:
            line = line.strip().split(';')
            for x in line:
                name, value = x.split('^')
                print("name =", name, " value=", value)
                if name in input_datas:
                    input_datas[name].append(value)
                else:
                    input_datas[name] = [value]

        fd.close()
        for k, v in input_datas.items():
          print("k=", k, " v=", v)
          if len(v) != size_info[k]:
            input_datas[k] = [input_datas[k]  + [''] * (int(size_info[k]) - len(v))]
          else:
            input_datas[k] = [input_datas[k]]
        print(input_datas['ag'])
        print(input_datas['ps'])
        print(input_datas)
        return input_datas

    input_data = handle_feature()
    print(input_data)
    tic = timeit.default_timer()
    predictions = predict_fn(input_data)
    toc = timeit.default_timer()
    print(predictions)
    print('predict time:', toc - tic)

#string_placeholder_input_predict_test(predict_fn)
string_placeholder_input_predict(predict_fn)
#string_decode_input_predict(predict_fn)
