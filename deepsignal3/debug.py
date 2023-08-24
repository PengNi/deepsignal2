import numpy as np
import pod5 as p5


def extract_signal_from_pod5(pod5_path):
    # signals = {}
    num = 0
    flag = 0
    with p5.Reader(pod5_path) as reader:
        for read_record in reader.reads():
            # signals[str(read_record.read_id)] = {
            #    "signal": read_record.signal,
            #    "shift": read_record.calibration.offset,
            #    "scale": read_record.calibration.scale,
            # }  # 不加str会变成UUID，很奇怪
            num += 1
            flag += 1
            if flag == 1000000:
                flag = 0
                print('现在统计了{}条reads'.format(num))
    print('pod5 has {} reads'.format(num))
    return num  # output.pod5 1581750#说明该文件转化不完全
# 20221109_1654_5A_PAG65784_f306681d all.pod5 has 5887455 reads
# 20221109_1654_5D_PAG68757_39c39833 all.pod5 has 6568755 reads
# merge 12456210 reads from 2 files
#pod5_path = "/homeb/xiaoyf/data/HG002/20221109_1654_5D_PAG68757_39c39833/pod5/all.pod5"
#num = extract_signal_from_pod5(pod5_path)
def statistics(filename):
    total_data = -1
    with open(filename, "r") as f:
        total_data = len(f.readlines())
    print(str(filename)+' has {} line'.format(total_data))#30000000

statistics('/home/xiaoyifu/data/HG002/R9.4/samples_CG.hc_poses.r30m.tsv')
