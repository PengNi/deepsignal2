deepsignal2
===========


Release
-------
0.1.4
-----
optimize/test more on multi-gpu support in call_mods module

add scipy==1.7.3 in requirements (floating point error in scipy>=1.9->tombo). fix this in deepsignal2, although tombo only is the pre-process tool for this tool.

replace np.int/np.float with int/float in extract_features module


0.1.3
-----
bug fixes in train module

multi-gpu support in call_mods module

updates dependences/requirements


0.1.2
-----
make sure results of each read be written together in call_mods' output

make `--reference_path` not a required input in *extract* and *call_mods* module


0.1.1
-----
add `init_model` option in train module,

fix bug of extrating contig name from fast5s


0.1.0
-----
Release the first vesrion of deepsignal2