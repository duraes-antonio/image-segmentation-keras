�	��0x@��0x@!��0x@	*~-���?*~-���?!*~-���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��0x@�ׂ��m@1�k*qa@A��2�,�?I.���@Y˽��P��?*	ʡE����@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator�jIyL@!!��Au�X@)�jIyL@1!��Au�X@:Preprocessing2F
Iterator::Model[���<�?!9:v�nz�?)���=^�?1�*� f��?:Preprocessing2P
Iterator::Model::Prefetchʤ�6 �?!�IM�v?�?)ʤ�6 �?1�IM�v?�?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap���yL@!r�]d��X@)ZF�=��n?10Ħ{?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 61.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9+~-���?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ׂ��m@�ׂ��m@!�ׂ��m@      ��!       "	�k*qa@�k*qa@!�k*qa@*      ��!       2	��2�,�?��2�,�?!��2�,�?:	.���@.���@!.���@B      ��!       J	˽��P��?˽��P��?!˽��P��?R      ��!       Z	˽��P��?˽��P��?!˽��P��?JGPUY+~-���?b �"n
Dgradient_tape/functional_23/block1_conv2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteruZ$A9b�?!uZ$A9b�?"A
functional_23/block1_conv2/Relu_FusedConv2Du����?!�프��?"l
Cgradient_tape/functional_23/block1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInputt9E��?!u2�UE)�?"i
@gradient_tape/functional_23/conv2d_13/Conv2D/Conv2DBackpropInputConv2DBackpropInput��1��{�?!��/D��?"i
@gradient_tape/functional_23/conv2d_11/Conv2D/Conv2DBackpropInputConv2DBackpropInput0N�e���?!�n��?"A
functional_23/block2_conv2/Relu_FusedConv2De����?!�t��*��?"l
Cgradient_tape/functional_23/block2_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInputz�!���?!��^��?"i
@gradient_tape/functional_23/conv2d_12/Conv2D/Conv2DBackpropInputConv2DBackpropInput\
�l1��?!<���>�?"l
Cgradient_tape/functional_23/block3_conv3/Conv2D/Conv2DBackpropInputConv2DBackpropInput�$��Ǔ?!�we��?"l
Cgradient_tape/functional_23/block3_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput�Cy:Ǔ?!I�\��/�?I���#�?Q������X@Y�{��F�,@a��k'�aU@q��@��)@y���� Z�?"�	
both�Your program is POTENTIALLY input-bound because 61.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�12.8824% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 