	Di�w@Di�w@!Di�w@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Di�w@�^|�ck@1�٭e2�a@A1�Z{���?IK�H���@*	��� �]�@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator[���O@!)�Z���X@)[���O@1)�Z���X@:Preprocessing2F
Iterator::Model�2�FY�?!zZ"��Ų?)��@��?1�%E�@�?:Preprocessing2P
Iterator::Model::Prefetch��	���?!��[��?)��	���?1��[��?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap��QcBO@!i�M�N�X@){O崧�l?1\���:w?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 59.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�^|�ck@�^|�ck@!�^|�ck@      ��!       "	�٭e2�a@�٭e2�a@!�٭e2�a@*      ��!       2	1�Z{���?1�Z{���?!1�Z{���?:	K�H���@K�H���@!K�H���@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb 