import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        type=str,
        default="/home/xiaoyf/methylation/deepsignal/log/processdata.log",
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
        default="/homeb/xiaoyf/data/HG002/bam/dorado/has_moves_chm13v2.bam",
        help="bam file store address.",
    )
    parser.add_argument(
        "--nproc", type=int, default=38, help="minimum number of processes in extract features."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="size of batch in extract features."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=21,
        help="size of window in extract neighbor features for target base.",
    )
    parser.add_argument(
        "--conv-in", type=int, default=4, help="Input sequence features"
    )
  
    parser.add_argument("--step-interval", type=int, default=100, required=False)
    parser.add_argument("--lr", type=float, default=0.001, required=False)
    parser.add_argument(
        "--train-file",
        type=str,
        help="feature file used in trainning",
        default="/homeb/xiaoyf/data/HG002/R9.4/samples_CG.hc_poses.r30m.tsv",
    )
    parser.add_argument(
        "--model-dir", type=str, default="/home/xiaoyf/methylation/deepsignal/log/"
    )
    parser.add_argument(
        "--target-chr", type=str, default="chr11"
    )
    parser.add_argument(
        "--max-epoch-num",
        action="store",
        default=10,
        type=int,
        required=False,
        help="max epoch num, default 10",
    )
    parser.add_argument(
        "--min-epoch-num",
        action="store",
        default=5,
        type=int,
        required=False,
        help="min epoch num, default 5",
    )
    parser.add_argument('--init_model', type=str, default=None, required=False,
                             help="file path of pre-trained model parameters to load before training")
    parser.add_argument('--tseed', type=int, default=1234,
                             help='random seed for pytorch')
    parser.add_argument("--nodes", default=1, type=int,
                              help="number of nodes for distributed training, default 1")
    parser.add_argument("--ngpus_per_node", default=2, type=int,
                              help="number of GPUs per node for distributed training, default 2")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12315", type=str,
                              help="url used to set up distributed training")
    parser.add_argument("--node_rank", default=0, type=int,
                              help="node rank for distributed training, default 0")
    parser.add_argument("--epoch_sync", action="store_true", default=False,
                              help="if sync model params of gpu0 to other local gpus after per epoch")
    return parser.parse_args()  # 在jupyter里面运行时()要加[]，在命令行运行时要去掉[]，不然都会报错
