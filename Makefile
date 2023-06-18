deps:
	pip install timm opencv-python absl-py imutils

watch:
	rg -t py --files | entr -s "black ./experiments.py; clear; python ./experiments.py"                      

get_weights:
	# cd weights && wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt 
	cd weights && wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt
	cd weights && wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt
	cd weights && wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt
	cd weights && wget https://github.com/isl-org/MiDaS/releases/download/v3_1/openvino_midas_v21_small_256.xml
	cd weights && wget https://github.com/isl-org/MiDaS/releases/download/v3_1/openvino_midas_v21_small_256.bin
