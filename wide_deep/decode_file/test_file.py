import tensorflow as tf
#decode_file_op_module = tf.load_op_library('./decode_file_op.so')
#print dir(decode_file_op_module)
decode_file_serve_op_module = tf.load_op_library('./decode_file_serve_op.so')
print dir(decode_file_serve_op_module)
#print help(decode_file_op_module)

#decode_file(records, record_skip, record_defaults, field_delim=None, field_outer_delim=None, 
# field_inner_delim=None, use_quote_delim=None, na_value=None, name=None)

#output_size = [1,1,3]
input_examples = [["0,1,2%3%5"], ["111,222,333%444"]]
#input_examples = ["0,1,2%3%5%8%9"]
#input_examples = ["0,1,2%3%5"]
output_size = [1, 1, 10]
record_default = [[0],[''],['']]
#features = decode_file_op_module.decode_file(input_examples, record_default,
#        output_size=output_size, field_outer_delim=',', field_inner_delim='%')
features = decode_file_serve_op_module.decode_file_serve(input_examples, record_default,
        output_size=output_size, field_outer_delim=',', field_inner_delim='%')
print('===features===', features)

with tf.Session('') as sess:
    print("=========== run =========")
    tmp = sess.run([features])
    print tmp
    print 'len:', len(tmp[0])
