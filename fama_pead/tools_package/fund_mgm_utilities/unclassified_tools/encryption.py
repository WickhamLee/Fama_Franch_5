import warnings
import hashlib


def get_func_input(output_from_locals, output_form_signature):
    function_input_dict = dict()
    for key in output_form_signature.parameters.keys():
        function_input_dict[key] = output_from_locals[key]
    return function_input_dict


def hash_dict(input_dict):
    key_str = ''
    value_str = ''
    for key in input_dict:
        key_str = key_str + key
        try:
            value_str = value_str + str(input_dict[key])
        except:
            data_type =  type(input_dict[key])
            data_type = data_type.__name__
            value_str = value_str + 'str_conversion_failed, data type is ' + data_type
            warnings.warn('hash_dict: 该key的值无法被转换成String： ' + key + '。其数据类型为：' + data_type + '. 签名结果可能不唯一。它的值是')

    hash_1 = hashlib.sha1(key_str.encode()).hexdigest() + hashlib.sha1(value_str.encode()).hexdigest()
    hash_value = hashlib.sha1(hash_1.encode()).hexdigest()
    return hash_value

def hash_function_input(output_from_locals, output_form_signature):
    function_input_dict = get_func_input(output_from_locals, output_form_signature)
    hash_value =  hash_dict(function_input_dict)
    return hash_value



def out_hash_value(key_str):
    hash_value = hashlib.sha1(key_str.encode()).hexdigest()
    return hash_value


def hash_file(file_path):
    import hashlib
    BLOCKSIZE = 65536
    hasher = hashlib.sha1()
    with open(file_path, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()
    
    