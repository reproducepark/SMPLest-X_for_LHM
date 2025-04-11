Guide to HumanData and View tools
========================

**What is HumanData?**
- 
HumanData is designed to provide a unified format for SMPL/SMPLX datasets to support joint training and evaluation.

The project is maintained in MMHuman3D.
See [detailed info](https://github.com/open-mmlab/mmhuman3d/blob/convertors/docs/human_data.md) for data structure and sample usage.

If you want to create your own humandata file, please refer to the sample below and maintain the similiar structure. Basically it is a big dictionary with some lists or dicts of lists, any dict with the correct structure works (Not necesscarily in `HumanData` class).

**Sample Visualize Script**
- 
We provide a simple script to check the annotation and visualize the results. The script will read the annotation from HumanData and render it on the corresponding image using pyrender.

Download sample here and extract to somewhere: [Hugging Face](https://huggingface.co/waanqii/SMPLest-X/resolve/main/hd_sample_SMPLestX.zip?download=true)

Rendered image will be saved in the output folder.
```
# <your path>: the path where you extract the sample
python humandata_prep/check.py \ 
    --hd_path <your path>/hd_10sample.npz \ 
    --image_folder <your path> \ 
    --output_folder <your path>/output \ 
    --body_model_path <your smplx body model path>
```
For 3d rendering, we suggest open3d.

**Important Points: when visualizing other humandata files** 
-
- Check `flat_hand_mean` if is correctly set, for humandata, it shoule be specified in `hd['misc']['flat_hand_mean']` or by default `False`
- Check `gender`
- For some specific datasets, they might provide mesh vertices instead of SMPL/SMPLX parameters, we suggest to fit the mesh to parameters for every instance to maintain the consistency of the visualization. Some of those datasets are:
    - Arctic: They provide `vtemplate` instead of `betas`
    - EHF: They provide mesh files
- Standalone [SMPLX parameters fitting script](https://github.com/open-mmlab/mmhuman3d/blob/convertors/tools/preprocess/fit_shape2smplx.py)
 


