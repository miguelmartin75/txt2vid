from txt2vid.data import my_dataset
from txt2vid.util.stopwatch import Stopwatch

#from nvidia.dali.plugin.pytorch import DALIGenericIterator

def get_loader(args):
    dset = my_dataset(data=args.data, use_lmdb=args.lmdb, vocab=args.vocab, anno=args.anno)
    return dset

def main(args):
    loader = get_loader(args)

    print("Num datapoints = %d" % len(loader))

    total_time = Stopwatch()
    avg_time_watch = Stopwatch()
    avg_time = 0

    total_time.start()
    avg_time_watch.start()
    for i, data in enumerate(loader):
        avg_time_watch.stop()
        avg_time += avg_time_watch.elapsed_time
        avg_time_watch.start()

    avg_time /= (i + 1)

    total_time.stop()
    print("Took: %.5f seconds (average time per iter = %.5f s)" % (total_time.elapsed_time, avg_time))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='input data')
    parser.add_argument('--lmdb', default=False, action='store_true', help='is the data LMDB?')
    parser.add_argument('--anno', type=str, default=None, help='annotation')
    parser.add_argument('--vocab', type=str, default=None, help='annotation')
    args = parser.parse_args()

    main(args)
