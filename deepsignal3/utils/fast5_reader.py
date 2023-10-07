"""Classes to interact with Fast5 files"""
import h5py
import numpy as np
import sys

reads_group_single = 'Raw/Reads'


class SingleFast5:
    def __init__(self, path, is_single=False, readname="", mode='r'):
        self.is_single = is_single
        self._readid = ""
        if is_single:
            self._read = h5py.File(path, mode=mode)
        else:
            self._read = path  # from multi-read format
            if readname.startswith("read_"):
                self._readid = readname[5:]
            else:
                self._readid = readname

    def close(self):
        if self.is_single:
            self._read.close()

    def get_readid(self):
        if self._readid != "":
            return self._readid
        first_read = list(self._read[reads_group_single].keys())[0]
        if sys.version_info[0] >= 3:
            try:
                read_id = str(self._read['/'.join([reads_group_single, first_read])].attrs['read_id'], 'utf-8')
            except TypeError:
                read_id = str(self._read['/'.join([reads_group_single, first_read])].attrs['read_id'])
        else:
            read_id = str(self._read['/'.join([reads_group_single, first_read])].attrs['read_id'])
        # print(read_id)
        return read_id

    def get_raw_signal(self):
        if self.is_single:
            first_read = list(self._read[reads_group_single].keys())[0]
            raw_dat = self._read['/'.join([reads_group_single, first_read, 'Signal'])][()]
        else:
            raw_dat = self._read['Raw/Signal'][()]
        return np.asarray(raw_dat)

    def get_lastest_basecallgroup(self):
        groups = [y for y in filter(lambda x: x.startswith('Basecall_1D'), self._read['Analyses'].keys())]
        groups = sorted(groups)
        return groups[-1]

    def get_fastq(self, basecall_group='Basecall_1D_000', basecall_subgroup='BaseCalled_template'):
        path = 'Analyses/{group}/{subgroup}/Fastq'.format(group=basecall_group,
                                                          subgroup=basecall_subgroup)
        return self._read[path][()].decode('UTF-8')

    def get_seq(self, basecall_group='Basecall_1D_000', basecall_subgroup='BaseCalled_template'):
        return self.get_fastq(basecall_group, basecall_subgroup).split('\n')[1].strip()

    def get_move(self, basecall_group='Basecall_1D_000', basecall_subgroup='BaseCalled_template'):
        path = 'Analyses/{group}/{subgroup}/Move'.format(group=basecall_group,
                                                         subgroup=basecall_subgroup)
        return np.asarray(self._read[path])

    def get_trace(self, basecall_group='Basecall_1D_000', basecall_subgroup='BaseCalled_template'):
        """Every trace are 8 values. That represents each the property of one of the bases
        (4 bases with each flip flop)
        The Trace order is: A, C, G, T/U, A, C, G, T/U.
        """
        path = 'Analyses/{group}/{subgroup}/Trace'.format(group=basecall_group,
                                                          subgroup=basecall_subgroup)
        return np.asarray(self._read[path])

    def get_trace_4(self, basecall_group='Basecall_1D_000', basecall_subgroup='BaseCalled_template'):
        trace = self.get_trace(basecall_group, basecall_subgroup)
        ncol = trace.shape[1]
        assert (ncol == 4 or ncol == 8)
        if ncol == 8:
            trace = trace[:, :4] + trace[:, 4:]
        return trace

    def get_stride(self, basecall_group='Basecall_1D_000'):
        path = 'Analyses/{group}/Summary/basecall_1d_template'.format(group=basecall_group)
        return int(self._read[path].attrs["block_stride"])

    def get_seqlen_from_groupattri(self, basecall_group='Basecall_1D_000'):
        path = 'Analyses/{group}/Summary/basecall_1d_template'.format(group=basecall_group)
        return int(self._read[path].attrs["sequence_length"])

    def get_basecallgroup_related_sementation(self, basecall_group='Basecall_1D_000'):
        path = 'Analyses/{group}'.format(group=basecall_group)
        if sys.version_info[0] >= 3:
            try:
                segmentation = str(self._read[path].attrs['segmentation'], 'utf-8')
            except TypeError:
                segmentation = str(self._read[path].attrs['segmentation'])
        else:
            segmentation = str(self._read[path].attrs['segmentation'])
        return segmentation

    def get_segmentation_start(self, segmentation_group="Segmentation_000"):
        path = 'Analyses/{group}/Summary/segmentation'.format(group=segmentation_group)
        return int(self._read[path].attrs["first_sample_template"])

    def get_segmentation_duration(self, segmentation_group="Segmentation_000"):
        path = 'Analyses/{group}/Summary/segmentation'.format(group=segmentation_group)
        return int(self._read[path].attrs["duration_template"])

    def check_signallen_against_segmentation(self, segmentation_group="Segmentation_000"):
        signal_len = len(self.get_raw_signal())
        assert signal_len == self.get_segmentation_start(segmentation_group) + \
            self.get_segmentation_duration(segmentation_group)

    def check_fastq_seqlen(self, basecall_group='Basecall_1D_000', basecall_subgroup='BaseCalled_template'):
        seqlen = len(self.get_seq(basecall_group, basecall_subgroup))
        assert seqlen == self.get_seqlen_from_groupattri(basecall_group)

    # https://github.com/nanoporetech/ont_fast5_api/blob/master/ont_fast5_api/fast5_read.py#L523
    def _get_scaling_of_a_read(self):
        global_key = "UniqueGlobalKey/"
        try:
            channel_info = dict(list(self._read[global_key + 'channel_id'].attrs.items()))
            digi = channel_info['digitisation']
            parange = channel_info['range']
            offset = channel_info['offset']
            scaling = float(parange) / digi

            return scaling, offset
        except Exception:
            return None, None

    def rescale_signals(self, signals):
        scaling, offset = self._get_scaling_of_a_read()
        if scaling is not None:
            return np.array(scaling * (signals + offset), dtype=float)
        else:
            return signals

    # map info ===
    def get_map_info(self, corrected_group, basecall_subgroup):
        alignment = self._read['Analyses/' + corrected_group + '/' + basecall_subgroup + '/Alignment']

        grp_cigar = alignment['cigar']
        cigars = [cigar.decode('UTF-8') for cigar in grp_cigar[()]]

        grp_chrom_strands = alignment['mapped_chrom']
        chrom_strands = []
        for chrom_strand in grp_chrom_strands:
            chrom, strand = chrom_strand[0].decode('UTF-8'), chrom_strand[1].decode('UTF-8')
            chrom_strands.append((chrom, strand))

        grp_fraginfo = alignment['mapped_frag']
        frags = []
        for frag in grp_fraginfo:
            # 'read_start', 'read_end', 'mapped_start', 'mapped_end', 'clipped_bases_start',
            # 'clipped_bases_end', 'num_insertion', 'num_deletions', 'num_matches',
            # 'num_mismatches'
            frags.append(tuple(frag[:4]))
        # cigar, chrom, strand, read_s, read_e, ref_s, ref_e
        return [(x,) + y + z for x, y, z in zip(cigars, chrom_strands, frags)]


# https://github.com/Fabianexe/SlaPPy/blob/master/slappy/fast5/__init__.py
class MultiFast5:
    """The class that represents one multi fast5 file"""

    def __init__(self, path, mode='r'):
        """Open the file.
        The file can be open in read (r), append(a) or write(w) modus.
        However, no writing options are implemented so far.

        :param path: The path to the file as string.
        :param mode: The mode how the file should open as one character string.(Default "r")
        """
        self.root = h5py.File(path, mode=mode)

    def close(self):
        self.root.close()

    def __iter__(self):
        """Iterate over the reads in the file

        :return: An iterator over the names of the reads.
        """
        for read_name in self.root:
            yield read_name

    def __getitem__(self, item):
        """Get for a name of a read the coresponding Fast5Read object

        :param item: The name of a read in the fast5 file as string
        :return: The group
        """
        return self.root[item]
