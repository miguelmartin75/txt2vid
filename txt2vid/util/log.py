# TODO: output to /dev/stderr for warn/err
# colour and timestamp
# 
# maybe use python logging module

def readable_timestamp():
    import datetime
    return datetime.datetime.now()

def status(msg):
    print('%s [STATUS]: %s' % (readable_timestamp(), msg))

def warn(msg):
    print('%s [WARNING]: %s' % (readable_timestamp(), msg))

def error(msg):
    print('%s [ERROR]: %s' % (readable_timestamp(), msg))
