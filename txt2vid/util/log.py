# TODO: output to /dev/stderr for warn/err
# colour and timestamp
# 
# maybe use python logging module

def status(msg):
    print('[STATUS]: %s' % msg)

def warn(msg):
    print('[WARNING]: %s' % msg)

def error(msg):
    print('[ERROR]: %s' % msg)
