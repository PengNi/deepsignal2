import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        type=str,
        default="/home/xiaoyf/methylation/deepsignal/log/process.log",
        help="log store address.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/home/xiaoyf/methylation/deepsignal/log/feature_chm13v2.txt",
        help="feature file store address.",
    )
    parser.add_argument(
        "--pod5-file",
        type=str,
        default="/homeb/xiaoyf/data/HG002/20221109_1654_5A_PAG65784_f306681d/pod5/output.pod5",
        help="pod5 file store address.",
    )
    parser.add_argument(
        "--bam-file",
        type=str,
        default="/homeb/xiaoyf/data/HG002/bam/has_moves_chm13v2.bam",
        help="bam file store address.",
    )
    parser.add_argument(
        "--nproc", type=int, default=400, help="number of processes in extract features."
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
