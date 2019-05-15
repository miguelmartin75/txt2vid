import torch

# https://stackoverflow.com/questions/452969/does-python-have-an-equivalent-to-java-class-forname
def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m

def create_object(json_obj, **kwargs):
    params = json_obj
    clz = get_class(params['class'])

    if 'args' in params:
        args = params['args']
        args.update(kwargs)
        return clz(**args)

    return clz()

# creates an object from a json file
# assumes json is of the form:
#
# { "class": <class-name>, "args": { "foo": 1, "bar": "abc" } }
#
# if args is not provided, will default construct. Otherwise,
# args is passed as kwargs to ctor.
#
# use kwargs to provide additional arguments
def create_object_file(json_file_path, **kwargs):
    import json
    with open(json_file_path) as in_f:
        params = json.load(in_f)
    assert('class' in params)

    return create_object(params, **kwargs)

if __name__ == '__main__':
    obj = {'class': 'txt2vid.models.tgan.discrim.Discrim',\
            'args': { \
                'sequence_first': False,\
                'in_channels': 2\
             }}
    obj = create_object(obj, mid_ch=90)
    print(obj)
