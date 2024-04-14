=> Final test
##### Evaluating market1501 (source) #####
Extracting features from query set ...
Done, obtained 3368-by-512 matrix
Extracting features from gallery set ...
Done, obtained 15913-by-512 matrix
Speed: 0.0129 sec/batch
Computing distance matrix with metric=euclidean ...
Computing CMC and mAP ...
** Results **
mAP: 25.6%
CMC curve
Rank-1  : 43.6%
Rank-5  : 67.3%
Rank-10 : 76.1%
Rank-20 : 84.1%
Checkpoint saved to "log/osnet/model/model.pth.tar-100"
Elapsed 1:35:43

=> Final test
##### Evaluating market1501 (source) #####
Extracting features from query set ...
Done, obtained 3368-by-512 matrix
Extracting features from gallery set ...
Done, obtained 15913-by-512 matrix
Speed: 0.0230 sec/batch
Computing distance matrix with metric=euclidean ...
Computing CMC and mAP ...
** Results **
mAP: 47.2%
CMC curve
Rank-1  : 68.4%
Rank-5  : 86.2%
Rank-10 : 91.2%
Rank-20 : 94.6%
Checkpoint saved to "log/osnet\model\model.pth.tar-300"
Elapsed 5:41:53

resnet18						Parameters: 11176512			FLOPs:1184366592
resnet34						Parameters: 21284672			FLOPs:2392326144
resnet50						Parameters: 23508032			FLOPs:2669150208
resnet101						Parameters: 42500160			FLOPs:5093457920
resnet152						Parameters: 58143808			FLOPs:7517765632
resnext50_32x4d						Parameters: 22979904			FLOPs:2761424896
resnext101_32x8d					Parameters: 86742336			FLOPs:10718019584
resnet50_fc512						Parameters: 24558144			FLOPs:4054319616
se_resnet50						Parameters: 26039024			FLOPs:2520686256
se_resnet50_fc512					Parameters: 27089136			FLOPs:3956187312
se_resnet101						Parameters: 47277872			FLOPs:4947240688
se_resnext50_32x4d					Parameters: 25510896			FLOPs:2763955888
se_resnext101_32x4d					Parameters: 46906416			FLOPs:5208336112
densenet121						Parameters: 6953856			FLOPs:1850212352
densenet169						Parameters: 12484480			FLOPs:2193096704
densenet201						Parameters: 18092928			FLOPs:2801270784
densenet161						Parameters: 26472000			FLOPs:5045354496
densenet121_fc512					Parameters: 7479680			FLOPs:1850737152
inceptionresnetv2					Parameters: 54306464			FLOPs:3770831584
inceptionv4						Parameters: 41142816			FLOPs:3609714528
xception						Parameters: 20806952			FLOPs:2963461536
resnet50_ibn_a						Parameters: 23508032			FLOPs:2669150208
resnet50_ibn_b						Parameters: 23509568			FLOPs:2669150208
nasnsetmobile						Parameters: 4232978			FLOPs:371319738
mobilenetv2_x1_0					Parameters: 2224960			FLOPs:203976704
mobilenetv2_x1_4					Parameters: 4291604			FLOPs:381847424
shufflenet						Parameters: 904728			FLOPs:89143296
squeezenet1_0						Parameters: 735424			FLOPs:469941120
squeezenet1_0_fc512					Parameters: 999104			FLOPs:470203776
squeezenet1_1						Parameters: 722496			FLOPs:168190272
shufflenet_v2_x0_5					Parameters: 341792			FLOPs:25764864
shufflenet_v2_x1_0					Parameters: 1253604			FLOPs:93965056
shufflenet_v2_x1_5					Parameters: 2478624			FLOPs:192480256
shufflenet_v2_x2_0					Parameters: 5344996			FLOPs:379562752
mudeep							Parameters: 139036280			FLOPs:3353842664
resnet50mid						Parameters: 27705408			FLOPs:2673345536
hacnn							model not match the input Input size does not match, expected (160, 64) but got (256, 128)
pcb_p6							Parameters: 23508032			FLOPs:4053270528
pcb_p4							Parameters: 23508032			FLOPs:4053270528
mlfn							Parameters: 32473024			FLOPs:2771421376
osnet_x1_0						Parameters: 2193616			FLOPs:978878352
osnet_x0_75						Parameters: 1299224			FLOPs:571754536
osnet_x0_5						Parameters: 636520			FLOPs:272901064
osnet_x0_25						Parameters: 203568			FLOPs:82316000
osnet_ibn_x1_0						Parameters: 2194640			FLOPs:978878352
osnet_ain_x1_0						Parameters: 2193616			FLOPs:978878352
osnet_ain_x0_75						Parameters: 1299224			FLOPs:571754536
osnet_ain_x0_5						Parameters: 636520			FLOPs:272901064
osnet_ain_x0_25						Parameters: 203568			FLOPs:82316000\

{'osnet_x0_25': 203568, 'osnet_ain_x0_25': 203568, 'shufflenet_v2_x0_5': 341792, 'osnet_x0_5': 636520, 'osnet_ain_x0_5': 636520, 'squeezenet1_1': 722496, 'squeezenet1_0': 735424, 'shufflenet': 904728, 'squeezenet1_0_fc512': 999104, 'shufflenet_v2_x1_0': 1253604, 'osnet_x0_75': 1299224, 'osnet_ain_x0_75': 1299224, 'osnet_x1_0': 2193616, 'osnet_ain_x1_0': 2193616, 'osnet_ibn_x1_0': 2194640, 'mobilenetv2_x1_0': 2224960, 'shufflenet_v2_x1_5': 2478624, 'nasnsetmobile': 4232978, 'mobilenetv2_x1_4': 4291604, 'shufflenet_v2_x2_0': 5344996, 'densenet121': 6953856, 'densenet121_fc512': 7479680, 'resnet18': 11176512, 'densenet169': 12484480, 'densenet201': 18092928, 'xception': 20806952, 'resnet34': 21284672, 'resnext50_32x4d': 22979904, 'resnet50': 23508032, 'resnet50_ibn_a': 23508032, 'pcb_p6': 23508032, 'pcb_p4': 23508032, 'resnet50_ibn_b': 23509568, 'resnet50_fc512': 24558144, 'se_resnext50_32x4d': 25510896, 'se_resnet50': 26039024, 'densenet161': 26472000, 'se_resnet50_fc512': 27089136, 'resnet50mid': 27705408, 'mlfn': 32473024, 'inceptionv4': 41142816, 'resnet101': 42500160, 'se_resnext101_32x4d': 46906416, 'se_resnet101': 47277872, 'inceptionresnetv2': 54306464, 'resnet152': 58143808, 'resnext101_32x8d': 86742336, 'mudeep': 139036280}

{'osnet_x0_25': 203568, 'osnet_ain_x0_25': 203568, 'shufflenet_v2_x0_5': 341792, 'osnet_x0_5': 636520, 'osnet_ain_x0_5': 636520, 'squeezenet1_1': 722496, 'squeezenet1_0': 735424, 'shufflenet': 904728, 'squeezenet1_0_fc512': 999104, 'shufflenet_v2_x1_0': 1253604, 'osnet_x0_75': 1299224, 'osnet_ain_x0_75': 1299224, 'osnet_x1_0': 2193616, 'osnet_ain_x1_0': 2193616, 'osnet_ibn_x1_0': 2194640, 'mobilenetv2_x1_0': 2224960, 'shufflenet_v2_x1_5': 2478624, 'nasnsetmobile': 4232978, 'mobilenetv2_x1_4': 4291604, 'shufflenet_v2_x2_0': 5344996, 'densenet121': 6953856, 'densenet121_fc512': 7479680, 'resnet18': 11176512, 'densenet169': 12484480, 'densenet201': 18092928, 'xception': 20806952, 'resnet34': 21284672, 'resnext50_32x4d': 22979904, 'resnet50': 23508032, 'resnet50_ibn_a': 23508032, 'pcb_p6': 23508032, 'pcb_p4': 23508032, 'resnet50_ibn_b': 23509568, 'resnet50_fc512': 24558144, 'se_resnext50_32x4d': 25510896, 'se_resnet50': 26039024, 'densenet161': 26472000, 'se_resnet50_fc512': 27089136, 'resnet50mid': 27705408, 'mlfn': 32473024, 'inceptionv4': 41142816, 'resnet101': 42500160, 'se_resnext101_32x4d': 46906416, 'se_resnet101': 47277872, 'inceptionresnetv2': 54306464, 'resnet152': 58143808, 'resnext101_32x8d': 86742336, 'mudeep': 139036280}

osnet_x0_25					Parameters: 203568
osnet_ain_x0_25					Parameters: 203568
shufflenet_v2_x0_5				Parameters: 341792
osnet_x0_5					Parameters: 636520
osnet_ain_x0_5					Parameters: 636520
squeezenet1_1					Parameters: 722496
squeezenet1_0					Parameters: 735424
shufflenet					Parameters: 904728
squeezenet1_0_fc512				Parameters: 999104
shufflenet_v2_x1_0				Parameters: 1253604
osnet_x0_75					Parameters: 1299224
osnet_ain_x0_75					Parameters: 1299224
osnet_x1_0					Parameters: 2193616
osnet_ain_x1_0					Parameters: 2193616
osnet_ibn_x1_0					Parameters: 2194640
mobilenetv2_x1_0				Parameters: 2224960
shufflenet_v2_x1_5				Parameters: 2478624
nasnsetmobile					Parameters: 4232978
mobilenetv2_x1_4				Parameters: 4291604
shufflenet_v2_x2_0				Parameters: 5344996
densenet121					Parameters: 6953856
densenet121_fc512				Parameters: 7479680
resnet18					Parameters: 11176512
densenet169					Parameters: 12484480
densenet201					Parameters: 18092928
xception					Parameters: 20806952
resnet34					Parameters: 21284672
resnext50_32x4d					Parameters: 22979904
resnet50					Parameters: 23508032
resnet50_ibn_a					Parameters: 23508032
pcb_p6						Parameters: 23508032
pcb_p4						Parameters: 23508032
resnet50_ibn_b					Parameters: 23509568
resnet50_fc512					Parameters: 24558144
se_resnext50_32x4d				Parameters: 25510896
se_resnet50					Parameters: 26039024
densenet161					Parameters: 26472000
se_resnet50_fc512				Parameters: 27089136
resnet50mid					Parameters: 27705408
mlfn						Parameters: 32473024
inceptionv4					Parameters: 41142816
resnet101					Parameters: 42500160
se_resnext101_32x4d				Parameters: 46906416
se_resnet101					Parameters: 47277872
inceptionresnetv2				Parameters: 54306464
resnet152					Parameters: 58143808
resnext101_32x8d				Parameters: 86742336
mudeep						Parameters: 139036280

osnet_x0_25					FLOPs: 203568
osnet_ain_x0_25					FLOPs: 203568
shufflenet_v2_x0_5				FLOPs: 341792
osnet_x0_5					FLOPs: 636520
osnet_ain_x0_5					FLOPs: 636520
squeezenet1_1					FLOPs: 722496
squeezenet1_0					FLOPs: 735424
shufflenet					FLOPs: 904728
squeezenet1_0_fc512				FLOPs: 999104
shufflenet_v2_x1_0				FLOPs: 1253604
osnet_x0_75					FLOPs: 1299224
osnet_ain_x0_75					FLOPs: 1299224
osnet_x1_0					FLOPs: 2193616
osnet_ain_x1_0					FLOPs: 2193616
osnet_ibn_x1_0					FLOPs: 2194640
mobilenetv2_x1_0				FLOPs: 2224960
shufflenet_v2_x1_5				FLOPs: 2478624
nasnsetmobile					FLOPs: 4232978
mobilenetv2_x1_4				FLOPs: 4291604
shufflenet_v2_x2_0				FLOPs: 5344996
densenet121					FLOPs: 6953856
densenet121_fc512				FLOPs: 7479680
resnet18					FLOPs: 11176512
densenet169					FLOPs: 12484480
densenet201					FLOPs: 18092928
xception					FLOPs: 20806952
resnet34					FLOPs: 21284672
resnext50_32x4d					FLOPs: 22979904
resnet50					FLOPs: 23508032
resnet50_ibn_a					FLOPs: 23508032
pcb_p6						FLOPs: 23508032
pcb_p4						FLOPs: 23508032
resnet50_ibn_b					FLOPs: 23509568
resnet50_fc512					FLOPs: 24558144
se_resnext50_32x4d				FLOPs: 25510896
se_resnet50					FLOPs: 26039024
densenet161					FLOPs: 26472000
se_resnet50_fc512				FLOPs: 27089136
resnet50mid					FLOPs: 27705408
mlfn						FLOPs: 32473024
inceptionv4					FLOPs: 41142816
resnet101					FLOPs: 42500160
se_resnext101_32x4d				FLOPs: 46906416
se_resnet101					FLOPs: 47277872
inceptionresnetv2				FLOPs: 54306464
resnet152					FLOPs: 58143808
resnext101_32x8d				FLOPs: 86742336
mudeep						FLOPs: 139036280


