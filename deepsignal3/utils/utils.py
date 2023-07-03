import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        type=str,
        default="/home/xiaoyf/methylation/deepsignal/log/test.log",
        help="log store address.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/home/xiaoyf/methylation/deepsignal/log/data.npy",
        help="feature file store address.",
    )
    parser.add_argument(
        "--pod5-file",
        type=str,
        default="/homeb/xiaoyf/data/HG002/example/pod5/output.pod5",
        help="pod5 file store address.",
    )
    parser.add_argument(
        "--bam-file",
        type=str,
        default="/homeb/xiaoyf/data/HG002/example/bam/has_moves.bam",
        help="bam file store address.",
    )
    parser.add_argument(
        "--nproc", type=int, default=4, help="number of processes in extract features."
    )
    parser.add_argument(
        "--batch-size", type=int, default=200, help="size of batch in extract features."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=21,
        help="size of window in extract neighbor features for target base.",
    )
    return parser.parse_args([])
